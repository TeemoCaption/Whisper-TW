from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = curr
    return prev[-1]


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    total_dist = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        total_dist += edit_distance(pred, ref)
        total_chars += len(ref)
    return total_dist / max(total_chars, 1)
