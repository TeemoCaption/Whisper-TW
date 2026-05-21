#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.config import load_config, resolve_device
from whisper_tw.char_vocab import CharacterVocab
from whisper_tw.metrics import character_error_rate, edit_distance
from whisper_tw.text_normalization import build_text_normalizer
from whisper_tw.training import build_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="評估 Whisper-TW 模型。")
    parser.add_argument("--config", required=True, help="設定檔路徑。")
    parser.add_argument("--checkpoint", help="模型 checkpoint 路徑。")
    parser.add_argument("--max-samples", type=int, help="只評估前 N 筆樣本。")
    parser.add_argument(
        "--output-dir",
        help="評估結果輸出目錄，預設使用 config 的 evaluation.output_dir 或 artifacts/eval。",
    )
    return parser.parse_args()


def resolve_eval_output_dir(args: argparse.Namespace, config: dict) -> Path:
    output_dir = (
        args.output_dir
        or config.get("evaluation", {}).get("output_dir")
        or "artifacts/eval"
    )
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_eval_json(output_dir: Path, payload: dict, filename: str) -> Path:
    output_path = output_dir / filename
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def synchronize_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(resolve_device(config))
    test_split = config["data"].get("test_split", "test")
    tokenizer, dataset, collator, model = build_components(
        config, test_split, args.max_samples
    )
    character_vocab = CharacterVocab.build_from_config(config)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("eval_batch_size", config["training"]["batch_size"])),
        shuffle=False,
        collate_fn=collator,
    )
    text_normalizer = build_text_normalizer(config["data"].get("text_normalization"))
    predictions: list[str] = []
    references: list[str] = []
    records: list[dict[str, object]] = []
    total_audio = 0
    total_edit_distance = 0
    total_reference_chars = 0
    batch_latencies: list[float] = []
    evaluation_cfg = config.get("evaluation", {})
    progress_update_every = max(int(evaluation_cfg.get("progress_update_every", 5)), 1)
    start = time.perf_counter()
    with torch.no_grad():
        total_batches = len(dataloader)
        progress = tqdm(
            dataloader,
            desc=f"eval {test_split}",
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress, start=1):
            input_features = batch["input_features"].to(device)
            synchronize_for_timing(device)
            batch_start = time.perf_counter()
            generated = model.generate_ctc_corrected(
                input_features=input_features,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
            )
            synchronize_for_timing(device)
            batch_elapsed = time.perf_counter() - batch_start
            batch_latencies.append(batch_elapsed)
            batch_predictions = [
                (
                    text_normalizer(character_vocab.decode(row.tolist()))
                    if text_normalizer.enabled
                    else character_vocab.decode(row.tolist())
                )
                for row in generated.cpu()
            ]
            predictions.extend(batch_predictions)
            references.extend(batch["texts"])
            for prediction, reference in zip(batch_predictions, batch["texts"]):
                sample_distance = edit_distance(prediction, reference)
                sample_reference_chars = len(reference)
                total_edit_distance += sample_distance
                total_reference_chars += sample_reference_chars
                records.append(
                    {
                        "reference": reference,
                        "prediction": prediction,
                        "char_error_rate": sample_distance / max(sample_reference_chars, 1),
                        "batch_inference_seconds": batch_elapsed,
                        "avg_seconds_per_sample_in_batch": batch_elapsed / max(len(batch_predictions), 1),
                    }
                )
            total_audio += input_features.size(0)
            if batch_index % progress_update_every == 0 or batch_index == total_batches:
                inference_elapsed = sum(batch_latencies)
                progress.set_postfix(
                    samples=len(references),
                    cer=f"{total_edit_distance / max(total_reference_chars, 1):.4f}",
                    sec_per_sample=f"{inference_elapsed / max(total_audio, 1):.3f}",
                )

    wall_clock_seconds = time.perf_counter() - start
    total_inference_seconds = sum(batch_latencies)
    cer = character_error_rate(predictions, references)
    output_dir = resolve_eval_output_dir(args, config)
    output_path = write_eval_json(
        output_dir,
        {
            "mode": "dataset_eval",
            "split": test_split,
            "checkpoint": args.checkpoint,
            "config_path": args.config,
            "samples": len(references),
            "cer": cer,
            "total_inference_seconds": total_inference_seconds,
            "seconds_per_sample": total_inference_seconds / max(total_audio, 1),
            "wall_clock_seconds": wall_clock_seconds,
            "non_inference_seconds": max(wall_clock_seconds - total_inference_seconds, 0.0),
            "batch_size": int(config["training"].get("eval_batch_size", config["training"]["batch_size"])),
            "batch_inference_seconds": batch_latencies,
            "records": records,
        },
        "eval.json",
    )
    print(f"split={test_split}")
    print(f"samples={len(references)}")
    print(f"cer={cer:.4f}")
    print(f"total_inference_seconds={total_inference_seconds:.3f}")
    print(f"seconds_per_sample={total_inference_seconds / max(total_audio, 1):.3f}")
    print(f"wall_clock_seconds={wall_clock_seconds:.3f}")
    print(f"eval_json={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
