#!/usr/bin/env python3
"""统一 ASR（Whisper）：音频 → 识别文本 → 逐字拼音，结果缓存。

本目录固定使用 Whisper，不随被评估模型变化。
产物: asr/cache/<音频名>.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pypinyin import Style, lazy_pinyin

CJK_RE = re.compile(r"[一-鿿]")
CACHE_DIR = Path(__file__).parent / "cache"


def extract_han(text: str) -> list[str]:
    return CJK_RE.findall(text)


def main() -> int:
    ap = argparse.ArgumentParser(description="Whisper 转写音频并输出逐字拼音（无声调）")
    ap.add_argument("audio", help="音频文件路径")
    ap.add_argument("--model", default="small",
                    help="Whisper 模型: tiny/base/small/medium (默认 small)")
    ap.add_argument("--language", default="zh", help="语言代码 (默认 zh)")
    ap.add_argument("--out", help="输出 json 路径 (默认 cache/<音频名>.json)")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"音频不存在: {audio}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else CACHE_DIR / f"{audio.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    import whisper
    model = whisper.load_model(args.model)
    hyp = model.transcribe(str(audio), language=args.language, fp16=False)["text"].strip()

    chars = [{"char": c,
              "pinyin": lazy_pinyin(c, style=Style.NORMAL)[0],
              "pinyin_tone": lazy_pinyin(c, style=Style.TONE)[0]}
             for c in extract_han(hyp)]

    payload = {
        "audio": str(audio.resolve()),
        "whisper_model": args.model,
        "text": hyp,
        "chars": chars,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"识别文本: {hyp}")
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
