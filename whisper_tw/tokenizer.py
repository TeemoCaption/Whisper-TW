from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


class SentencePieceTextTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {self.model_path}. "
                "Run scripts/train_tokenizer.py first."
            )
        self.processor.load(str(self.model_path))

    @property
    def pad_id(self) -> int:
        return self.processor.pad_id()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = list(self.processor.encode(text, out_type=int))
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        filtered = [
            token_id
            for token_id in ids
            if token_id not in {self.pad_id, self.bos_id, self.eos_id}
        ]
        return self.processor.decode(filtered)


def train_sentencepiece_from_corpus(
    corpus_path: str | Path,
    model_prefix: str | Path,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
) -> None:
    prefix = Path(model_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        normalization_rule_name="identity",
    )
