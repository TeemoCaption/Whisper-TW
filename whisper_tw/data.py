from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

from .bopomofo import BopomofoVocab
from .tokenizer import SentencePieceTextTokenizer


@dataclass(frozen=True)
class CommonVoiceSample:
    audio_path: Path
    text: str


def read_common_voice_split(data_root: str | Path, split: str) -> list[CommonVoiceSample]:
    root = Path(data_root)
    tsv_path = root / f"{split}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing Common Voice split: {tsv_path}")

    samples: list[CommonVoiceSample] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("sentence") or "").strip()
            rel_path = (row.get("path") or "").strip()
            if not text or not rel_path:
                continue
            samples.append(CommonVoiceSample(audio_path=root / "clips" / rel_path, text=text))
    return samples


class CommonVoiceTaiwanDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        text_tokenizer: SentencePieceTextTokenizer,
        bopomofo_vocab: BopomofoVocab,
        sample_rate: int = 16000,
        max_audio_seconds: float = 30.0,
        max_samples: int | None = None,
    ) -> None:
        self.samples = read_common_voice_split(data_root, split)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.text_tokenizer = text_tokenizer
        self.bopomofo_vocab = bopomofo_vocab
        self.sample_rate = sample_rate
        self.max_audio_samples = int(sample_rate * max_audio_seconds)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        waveform, sr = torchaudio.load(sample.audio_path)
        waveform = waveform.mean(dim=0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform[: self.max_audio_samples]
        return {
            "audio": waveform,
            "text": sample.text,
            "text_ids": torch.tensor(self.text_tokenizer.encode(sample.text), dtype=torch.long),
            "bopomofo_ids": torch.tensor(self.bopomofo_vocab.encode(sample.text), dtype=torch.long),
        }


class WhisperTWCollator:
    def __init__(
        self,
        feature_extractor: WhisperFeatureExtractor,
        text_pad_id: int,
        bopomofo_pad_id: int,
        sample_rate: int = 16000,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.text_pad_id = text_pad_id
        self.bopomofo_pad_id = bopomofo_pad_id
        self.sample_rate = sample_rate

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        audios = [item["audio"].numpy() for item in batch]
        features = self.feature_extractor(
            audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        text_ids = torch.nn.utils.rnn.pad_sequence(
            [item["text_ids"] for item in batch],
            batch_first=True,
            padding_value=self.text_pad_id,
        )
        bopomofo_ids = torch.nn.utils.rnn.pad_sequence(
            [item["bopomofo_ids"] for item in batch],
            batch_first=True,
            padding_value=self.bopomofo_pad_id,
        )
        return {
            "input_features": features.input_features,
            "decoder_input_ids": text_ids[:, :-1],
            "labels": text_ids[:, 1:],
            "bopomofo_labels": bopomofo_ids,
            "bopomofo_label_lengths": torch.tensor(
                [int((item["bopomofo_ids"] != self.bopomofo_pad_id).sum()) for item in batch],
                dtype=torch.long,
            ),
            "texts": [item["text"] for item in batch],
        }


def iter_tokenizer_sentences(data_root: str | Path, splits: list[str]):
    for split in splits:
        for sample in read_common_voice_split(data_root, split):
            yield sample.text
