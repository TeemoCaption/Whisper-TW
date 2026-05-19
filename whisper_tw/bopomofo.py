from __future__ import annotations

from dataclasses import dataclass

from pypinyin import Style, lazy_pinyin


BOPOMOFO_BLANK = "<blank>"
BOPOMOFO_PAD = "<pad>"
BOPOMOFO_UNK = "<unk>"


def text_to_bopomofo_units(text: str) -> list[str]:
    symbols = lazy_pinyin(text, style=Style.BOPOMOFO, errors=lambda chars: list(chars))
    units: list[str] = []
    for symbol in symbols:
        symbol = symbol.strip()
        if not symbol:
            continue
        units.extend(list(symbol))
    return units


@dataclass(frozen=True)
class BopomofoVocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    blank_id: int = 0
    pad_id: int = 1
    unk_id: int = 2

    @classmethod
    def default(cls) -> "BopomofoVocab":
        tokens = [
            BOPOMOFO_BLANK,
            BOPOMOFO_PAD,
            BOPOMOFO_UNK,
            "ㄅ",
            "ㄆ",
            "ㄇ",
            "ㄈ",
            "ㄉ",
            "ㄊ",
            "ㄋ",
            "ㄌ",
            "ㄍ",
            "ㄎ",
            "ㄏ",
            "ㄐ",
            "ㄑ",
            "ㄒ",
            "ㄓ",
            "ㄔ",
            "ㄕ",
            "ㄖ",
            "ㄗ",
            "ㄘ",
            "ㄙ",
            "ㄧ",
            "ㄨ",
            "ㄩ",
            "ㄚ",
            "ㄛ",
            "ㄜ",
            "ㄝ",
            "ㄞ",
            "ㄟ",
            "ㄠ",
            "ㄡ",
            "ㄢ",
            "ㄣ",
            "ㄤ",
            "ㄥ",
            "ㄦ",
            "ˉ",
            "ˊ",
            "ˇ",
            "ˋ",
            "˙",
        ]
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(token_to_id=token_to_id, id_to_token={idx: token for token, idx in token_to_id.items()})

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(unit, self.unk_id) for unit in text_to_bopomofo_units(text)]
