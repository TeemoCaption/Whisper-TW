#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.config import load_config
from whisper_tw.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="訓練 Whisper-TW 模型。")
    parser.add_argument("--config", required=True, help="設定檔路徑。")
    parser.add_argument("--max-samples", type=int, help="只載入前 N 筆樣本，方便煙霧測試。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    train(config, max_samples=args.max_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
