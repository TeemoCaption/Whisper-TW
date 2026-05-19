from __future__ import annotations

from pathlib import Path
from typing import Any

import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import WhisperFeatureExtractor

from .bopomofo import BopomofoVocab
from .config import resolve_device
from .data import CommonVoiceTaiwanDataset, WhisperTWCollator
from .model import WhisperTWModel
from .tokenizer import SentencePieceTextTokenizer


class WandbLogger:
    def __init__(self, config: dict[str, Any]) -> None:
        wandb_cfg = config.get("wandb", {})
        self.enabled = bool(wandb_cfg.get("enabled", False))
        self.run = None
        if not self.enabled:
            return

        import wandb

        mode = os.environ.get("WANDB_MODE") or str(wandb_cfg.get("mode", "online"))
        workspace_root = Path.cwd()
        wandb_root = workspace_root / "wandb"
        wandb_config_dir = workspace_root / ".wandb_config"
        wandb_cache_dir = workspace_root / ".wandb_cache"
        wandb_data_dir = workspace_root / ".wandb_data"
        for path in (wandb_root, wandb_config_dir, wandb_cache_dir, wandb_data_dir):
            path.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("WANDB_DIR", str(wandb_root))
        os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_config_dir))
        os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache_dir))
        os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data_dir))

        if mode == "online":
            api_key = os.environ.get("WANDB_API_KEY", "").strip()
            netrc_candidates = [
                Path.home() / ".netrc",
                Path.home() / "_netrc",
            ]
            if os.name == "nt":
                netrc_candidates.extend(
                    [
                        Path.home().parent / "_netrc",
                        Path.home().parent / ".netrc",
                    ]
                )
            has_netrc = any(path.exists() for path in netrc_candidates)
            if not api_key and not has_netrc:
                raise RuntimeError(
                    "wandb 已啟用為 online，但目前沒有可用的 WANDB_API_KEY。"
                    "請先執行 `wandb login`，或先設定 `WANDB_API_KEY`。"
                )

        settings = wandb.Settings(
            console=str(wandb_cfg.get("console", "redirect")),
            x_files_dir=str(wandb_root),
        )
        self.run = wandb.init(
            project=str(wandb_cfg.get("project", "Whisper-TW")),
            name=wandb_cfg.get("name"),
            mode=mode,
            config=config,
            dir=str(wandb_root),
            settings=settings,
        )

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        if not self.enabled:
            return
        import wandb

        wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def build_components(
    config: dict[str, Any], split: str, max_samples: int | None = None
):
    data_cfg = config["data"]
    tokenizer = SentencePieceTextTokenizer(config["tokenizer"]["model_path"])
    bopomofo_vocab = BopomofoVocab.default()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config["model"]["whisper_name"]
    )
    dataset = CommonVoiceTaiwanDataset(
        data_root=data_cfg["root"],
        split=split,
        text_tokenizer=tokenizer,
        bopomofo_vocab=bopomofo_vocab,
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        max_audio_seconds=float(data_cfg.get("max_audio_seconds", 30.0)),
        max_samples=max_samples,
    )
    collator = WhisperTWCollator(
        feature_extractor=feature_extractor,
        text_pad_id=tokenizer.pad_id,
        bopomofo_pad_id=bopomofo_vocab.pad_id,
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
    )
    model = WhisperTWModel(
        config=config,
        vocab_size=tokenizer.vocab_size,
        bopomofo_vocab_size=bopomofo_vocab.size,
        text_pad_id=tokenizer.pad_id,
    )
    return tokenizer, dataset, collator, model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def build_dataloader_kwargs(
    train_cfg: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(train_cfg.get("persistent_workers", num_workers > 0))
    dataloader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return dataloader_kwargs


def use_mixed_precision(train_cfg: dict[str, Any], device: torch.device) -> bool:
    return bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"


def get_amp_dtype(
    train_cfg: dict[str, Any], device: torch.device
) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    amp_dtype = str(train_cfg.get("amp_dtype", "auto")).lower()
    if amp_dtype == "original":
        return None
    if amp_dtype in {"auto", "default"}:
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if amp_dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if amp_dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if amp_dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")


def use_grad_scaler(amp_enabled: bool, amp_dtype: torch.dtype | None) -> bool:
    return amp_enabled and amp_dtype in {None, torch.float16}


def configure_training_runtime(train_cfg: dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    allow_tf32 = bool(train_cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(
            str(train_cfg.get("float32_matmul_precision", "high"))
        )


def autocast_context(
    device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype | None
):
    device_type = "cuda" if device.type == "cuda" else device.type
    autocast_kwargs: dict[str, Any] = {
        "device_type": device_type,
        "enabled": amp_enabled,
    }
    if amp_dtype is not None:
        autocast_kwargs["dtype"] = amp_dtype
    return torch.amp.autocast(**autocast_kwargs)


@torch.no_grad()
def val_loss(
    model: WhisperTWModel,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_text_loss = 0.0
    total_ctc_loss = 0.0
    total_batches = 0

    progress = tqdm(
        dataloader,
        desc="val",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, amp_enabled, amp_dtype):
            output = model(
                input_features=batch["input_features"],
                decoder_input_ids=batch["decoder_input_ids"],
                labels=batch["labels"],
                bopomofo_labels=batch["bopomofo_labels"],
                bopomofo_label_lengths=batch["bopomofo_label_lengths"],
            )
        if output.loss is None:
            raise RuntimeError("Validation loss is None.")
        total_loss += float(output.loss.detach().cpu())
        total_text_loss += (
            float(output.text_loss.detach().cpu())
            if output.text_loss is not None
            else 0.0
        )
        total_ctc_loss += (
            float(output.bopomofo_ctc_loss.detach().cpu())
            if output.bopomofo_ctc_loss is not None
            else 0.0
        )
        total_batches += 1
        progress.set_postfix(loss=f"{total_loss / total_batches:.4f}")

    model.train()
    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "text_loss": total_text_loss / denom,
        "bopomofo_ctc_loss": total_ctc_loss / denom,
    }


def train(config: dict[str, Any], max_samples: int | None = None) -> Path:
    train_cfg = config["training"]
    logger = WandbLogger(config)
    device = torch.device(resolve_device(config))
    configure_training_runtime(train_cfg, device)
    try:
        _, dataset, collator, model = build_components(
            config, config["data"].get("train_split", "train"), max_samples
        )
        val_dataset = CommonVoiceTaiwanDataset(
            data_root=config["data"]["root"],
            split=config["data"].get("dev_split", "dev"),
            text_tokenizer=SentencePieceTextTokenizer(
                config["tokenizer"]["model_path"]
            ),
            bopomofo_vocab=BopomofoVocab.default(),
            sample_rate=int(config["data"].get("sample_rate", 16000)),
            max_audio_seconds=float(config["data"].get("max_audio_seconds", 30.0)),
            max_samples=max_samples,
        )
        model.to(device)
        model.train()
        amp_enabled = use_mixed_precision(train_cfg, device)
        amp_dtype = get_amp_dtype(train_cfg, device)
        dataloader_kwargs = build_dataloader_kwargs(train_cfg, device)

        train_dataloader = DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            collate_fn=collator,
            **dataloader_kwargs,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
            shuffle=False,
            collate_fn=collator,
            **dataloader_kwargs,
        )
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        scaler = torch.amp.GradScaler(
            device=device.type,
            enabled=use_grad_scaler(amp_enabled, amp_dtype),
        )

        global_step = 0
        early_cfg = train_cfg.get("early_stopping", {})
        early_enabled = bool(early_cfg.get("enabled", True))
        patience = int(early_cfg.get("patience", 5))
        min_delta = float(early_cfg.get("min_delta", 0.001))
        epochs_without_improvement = 0
        best_val_loss = float("inf")
        best_checkpoint: Path | None = None
        for epoch in range(1, int(train_cfg.get("num_epochs", 1)) + 1):
            epoch_loss = 0.0
            epoch_batches = 0
            optimizer.zero_grad(set_to_none=True)
            print(f"epoch={epoch} phase=train")
            train_progress = tqdm(
                train_dataloader,
                desc=f"train {epoch}/{int(train_cfg.get('num_epochs', 1))}",
                leave=False,
                dynamic_ncols=True,
            )
            for batch in train_progress:
                batch = move_batch_to_device(batch, device)
                with autocast_context(device, amp_enabled, amp_dtype):
                    output = model(
                        input_features=batch["input_features"],
                        decoder_input_ids=batch["decoder_input_ids"],
                        labels=batch["labels"],
                        bopomofo_labels=batch["bopomofo_labels"],
                        bopomofo_label_lengths=batch["bopomofo_label_lengths"],
                    )
                if output.loss is None:
                    raise RuntimeError("Training loss is None.")
                scaler.scale(output.loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                epoch_loss += float(output.loss.detach().cpu())
                epoch_batches += 1
                text_loss = (
                    output.text_loss.item()
                    if output.text_loss is not None
                    else float("nan")
                )
                ctc_loss = (
                    output.bopomofo_ctc_loss.item()
                    if output.bopomofo_ctc_loss is not None
                    else float("nan")
                )
                train_progress.set_postfix(
                    step=global_step,
                    loss=f"{output.loss.item():.4f}",
                    early_stop=(
                        f"{epochs_without_improvement}/{patience}"
                        if early_enabled
                        else "off"
                    ),
                )
                logger.log(
                    {
                        "train/loss": float(output.loss.detach().cpu()),
                        "train/text_loss": float(text_loss),
                        "train/bopomofo_ctc_loss": float(ctc_loss),
                        "train/epoch": epoch,
                        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "train/mixed_precision": int(amp_enabled),
                    },
                    step=global_step,
                )
            train_loss = epoch_loss / max(epoch_batches, 1)
            val_metrics = val_loss(
                model,
                val_dataloader,
                device,
                amp_enabled,
                amp_dtype,
            )
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_text_loss={val_metrics['text_loss']:.4f} "
                f"val_bopomofo_ctc_loss={val_metrics['bopomofo_ctc_loss']:.4f}"
            )

            logger.log(
                {
                    "epoch/train_loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/text_loss": val_metrics["text_loss"],
                    "val/bopomofo_ctc_loss": val_metrics["bopomofo_ctc_loss"],
                    "val/epoch": epoch,
                },
                step=global_step,
            )

            improved = val_metrics["loss"] < (best_val_loss - min_delta)
            if improved:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                logger.log(
                    {"val/best_loss": best_val_loss, "val/best_epoch": epoch},
                    step=global_step,
                )
                best_checkpoint = save_checkpoint(
                    model,
                    config,
                    global_step,
                    epoch,
                    name="best",
                    metrics={"best_val_loss": best_val_loss, **val_metrics},
                )
            else:
                epochs_without_improvement += 1
                print(
                    f"epoch={epoch} early_stopping_no_improve="
                    f"{epochs_without_improvement}/{patience}"
                )
                logger.log(
                    {
                        "early_stopping/no_improve_epochs": epochs_without_improvement,
                        "early_stopping/patience": patience,
                    },
                    step=global_step,
                )

            if early_enabled and epochs_without_improvement >= patience:
                print(
                    f"early_stopping=triggered epoch={epoch} "
                    f"best_val_loss={best_val_loss:.4f}"
                )
                logger.log(
                    {
                        "early_stopping/triggered": 1,
                        "early_stopping/stopped_epoch": epoch,
                    },
                    step=global_step,
                )
                break

        final_checkpoint = save_checkpoint(
            model, config, global_step, epoch, name="last"
        )
        if best_checkpoint is None:
            return final_checkpoint
        return best_checkpoint
    finally:
        logger.finish()


def save_checkpoint(
    model: WhisperTWModel,
    config: dict[str, Any],
    step: int,
    epoch: int,
    name: str = "last",
    metrics: dict[str, float] | None = None,
) -> Path:
    output_dir = Path(config["training"].get("output_dir", "artifacts/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"whisper_tw_{name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": config,
            "metrics": metrics or {},
        },
        checkpoint_path,
    )
    print(f"checkpoint={checkpoint_path}")
    return checkpoint_path
