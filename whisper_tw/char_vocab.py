from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import resolve_common_voice_split_source
from .data import read_common_voice_split
from .text_normalization import build_text_normalizer


CHAR_BLANK = "<blank>"
CHAR_PAD = "<pad>"
CHAR_UNK = "<unk>"


@dataclass(frozen=True)
class CharacterVocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    blank_id: int = 0
    pad_id: int = 1
    unk_id: int = 2

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> "CharacterVocab":
        data_cfg = config["data"]
        char_cfg = config.get("character_vocab", {})
        vocab_path = char_cfg.get("path")
        if vocab_path and Path(vocab_path).exists():
            return cls.load(vocab_path)

        splits = list(char_cfg.get("splits") or data_cfg.get("tokenizer_text_splits") or [])
        if not splits:
            splits = [
                data_cfg.get("train_split", "train"),
                data_cfg.get("dev_split", "dev"),
                data_cfg.get("test_split", "test"),
            ]
        normalizer = build_text_normalizer(data_cfg.get("text_normalization"))
        characters: set[str] = set()
        for split in splits:
            split_source = resolve_common_voice_split_source(data_cfg, split)
            for sample in read_common_voice_split(data_cfg["root"], split_source):
                text = normalizer(sample.text) if normalizer.enabled else sample.text
                characters.update(ch for ch in text if ch.strip())

        tokens = [CHAR_BLANK, CHAR_PAD, CHAR_UNK, *sorted(characters)]
        vocab = cls(
            token_to_id={token: index for index, token in enumerate(tokens)},
            id_to_token={index: token for index, token in enumerate(tokens)},
        )
        if vocab_path:
            vocab.save(vocab_path)
        return vocab

    @classmethod
    def load(cls, path: str | Path) -> "CharacterVocab":
        tokens = Path(path).read_text(encoding="utf-8").splitlines()
        token_to_id = {token: index for index, token in enumerate(tokens)}
        return cls(
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
        )

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str | Path) -> None:
        vocab_path = Path(path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.write_text(
            "\n".join(self.id_to_token[index] for index in range(self.size)) + "\n",
            encoding="utf-8",
        )

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(ch, self.unk_id) for ch in text if ch.strip()]

    def decode(self, ids: list[int]) -> str:
        ignored = {self.blank_id, self.pad_id}
        return "".join(
            self.id_to_token.get(token_id, "")
            for token_id in ids
            if token_id not in ignored
        )
