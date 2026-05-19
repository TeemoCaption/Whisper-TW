from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import WhisperModel


@dataclass
class WhisperTWOutput:
    loss: torch.Tensor | None
    text_loss: torch.Tensor | None
    bopomofo_ctc_loss: torch.Tensor | None
    logits: torch.Tensor
    bopomofo_logits: torch.Tensor


class QFormer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, encoder_states: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_states.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return self.decoder(tgt=queries, memory=encoder_states)


class WhisperTWModel(nn.Module):
    def __init__(self, config: dict[str, Any], vocab_size: int, bopomofo_vocab_size: int, text_pad_id: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.text_pad_id = text_pad_id
        self.ctc_weight = float(model_cfg["bopomofo_ctc"].get("loss_weight", 0.3))
        self.qformer_enabled = bool(model_cfg["qformer"].get("enabled", True))

        self.whisper = WhisperModel.from_pretrained(model_cfg["whisper_name"])
        whisper_hidden = self.whisper.config.d_model
        if model_cfg.get("freeze_whisper_encoder", True):
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False

        hidden_size = int(model_cfg["decoder"]["hidden_size"])
        self.encoder_projection = nn.Linear(whisper_hidden, hidden_size)

        assist_cfg = model_cfg["assistant_encoder"]
        assist_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=int(assist_cfg["num_heads"]),
            dim_feedforward=hidden_size * 4,
            dropout=float(assist_cfg.get("dropout", 0.1)),
            batch_first=True,
            norm_first=True,
        )
        self.assistant_encoder = nn.TransformerEncoder(
            assist_layer,
            num_layers=int(assist_cfg["num_layers"]),
        )

        self.bopomofo_head = nn.Linear(hidden_size, bopomofo_vocab_size)
        self.bopomofo_aux = nn.Linear(bopomofo_vocab_size, hidden_size)

        q_cfg = model_cfg["qformer"]
        if self.qformer_enabled:
            self.qformer = QFormer(
                hidden_size=hidden_size,
                num_queries=int(q_cfg["num_queries"]),
                num_layers=int(q_cfg["num_layers"]),
                num_heads=int(q_cfg["num_heads"]),
                dropout=float(q_cfg.get("dropout", 0.1)),
            )
        else:
            self.qformer = None

        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=text_pad_id)
        self.position_embedding = nn.Embedding(int(model_cfg["decoder"]["max_target_length"]), hidden_size)
        dec_cfg = model_cfg["decoder"]
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=int(dec_cfg["num_heads"]),
            dim_feedforward=hidden_size * 4,
            dropout=float(dec_cfg.get("dropout", 0.1)),
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(dec_cfg["num_layers"]))
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        bopomofo_labels: torch.Tensor | None = None,
        bopomofo_label_lengths: torch.Tensor | None = None,
    ) -> WhisperTWOutput:
        encoder_hidden = self.whisper.encoder(input_features).last_hidden_state
        encoder_hidden = self.encoder_projection(encoder_hidden)
        encoder_hidden = self.assistant_encoder(encoder_hidden)

        bopomofo_logits = self.bopomofo_head(encoder_hidden)
        bopomofo_aux = self.bopomofo_aux(torch.softmax(bopomofo_logits, dim=-1))

        if self.qformer_enabled and self.qformer is not None:
            context = self.qformer(encoder_hidden + bopomofo_aux)
        else:
            context = encoder_hidden + bopomofo_aux

        token_states = self.token_embedding(decoder_input_ids)
        positions = torch.arange(decoder_input_ids.size(1), device=decoder_input_ids.device)
        positions = positions.unsqueeze(0).expand_as(decoder_input_ids)
        token_states = token_states + self.position_embedding(positions)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_input_ids.size(1),
            device=decoder_input_ids.device,
        )
        decoded = self.decoder(tgt=token_states, memory=context, tgt_mask=tgt_mask)
        logits = self.output_head(decoded)

        text_loss = None
        if labels is not None:
            text_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=self.text_pad_id,
            )

        ctc_loss = None
        if bopomofo_labels is not None and bopomofo_label_lengths is not None:
            log_probs = bopomofo_logits.log_softmax(dim=-1).transpose(0, 1)
            input_lengths = torch.full(
                size=(bopomofo_logits.size(0),),
                fill_value=bopomofo_logits.size(1),
                dtype=torch.long,
                device=bopomofo_logits.device,
            )
            ctc_loss = nn.functional.ctc_loss(
                log_probs=log_probs,
                targets=bopomofo_labels,
                input_lengths=input_lengths,
                target_lengths=bopomofo_label_lengths.to(bopomofo_logits.device),
                blank=0,
                zero_infinity=True,
            )

        loss = None
        if text_loss is not None:
            loss = text_loss
            if ctc_loss is not None:
                loss = loss + self.ctc_weight * ctc_loss

        return WhisperTWOutput(
            loss=loss,
            text_loss=text_loss,
            bopomofo_ctc_loss=ctc_loss,
            logits=logits,
            bopomofo_logits=bopomofo_logits,
        )

    @torch.no_grad()
    def generate_greedy(
        self,
        input_features: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        self.eval()
        generated = torch.full(
            (input_features.size(0), 1),
            bos_id,
            dtype=torch.long,
            device=input_features.device,
        )
        for _ in range(max_new_tokens):
            output = self(input_features=input_features, decoder_input_ids=generated)
            next_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_ids], dim=1)
            if bool((next_ids == eos_id).all()):
                break
        return generated
