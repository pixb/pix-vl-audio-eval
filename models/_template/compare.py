#!/usr/bin/env python3
"""评估大模型对原稿的理解：LLM 标注拼音 vs Whisper 音频读音拼音。

输入:
  understanding.json  被评估模型对原稿逐字标注的拼音（见 understanding.example.json）
  asr cache json      asr/transcribe.py 的产物（音频实际读音的拼音）

输出: 本目录 report/report.md 与 report.json
"""

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from pypinyin import pinyin
from rapidfuzz.distance import Levenshtein

CJK_RE = re.compile(r"[一-鿿]")
STRICT_TONE = False  # 由 CLI --strict-tone 控制


def extract_han(text: str) -> list[str]:
    return CJK_RE.findall(text)


def _audio_pinyin_field() -> str:
    return "pinyin_tone" if STRICT_TONE else "pinyin"


def norm_py(s: str) -> str:
    """拼音归一化：小写、去空格、ü→v；默认（非严格声调）时去掉声调符号。"""
    s = s.strip().lower().replace("ü", "v").replace(":", "")
    if not STRICT_TONE:
        s = "".join(c for c in unicodedata.normalize("NFD", s)
                    if not unicodedata.combining(c))
    return s


def pinyin_equiv(llm_py: str, audio_py: str, audio_char: str) -> bool:
    """模型理解的拼音 与 音频读音拼音 是否等价。

    音频侧拼音 = 识别字的词典读音（多音字默认取词典首位，如「佛」默认 fu），
    因此除直接比对外，模型给出的读音只要是识别字的合法读音也视为一致。
    """
    if norm_py(llm_py) == norm_py(audio_py):
        return True
    readings = {norm_py(r) for r in pinyin(audio_char, heteronym=True)[0]}
    return norm_py(llm_py) in readings


@dataclass
class DiffItem:
    kind: str  # wrong=理解偏差(读音不同) audio_missing=音频未覆盖 extra=音频多出
    ref_index: int | None      # 原稿字序号（1-based）
    ref_char: str | None
    llm_pinyin: str | None     # 被评估模型的理解
    audio_pinyin: str | None   # 音频实际读音（Whisper）
    audio_char: str | None


@dataclass
class CheckResult:
    model_label: str
    llm_text: str
    audio_text: str
    ref_han_count: int
    char_accuracy: float
    is_match: bool
    diffs: list[DiffItem] = field(default_factory=list)


def compare(und: dict, asr: dict, model_label: str = "") -> CheckResult:
    llm_chars = extract_han(und["text"])
    llm_py = list(und["pinyin"])
    if len(llm_chars) != len(llm_py):
        raise ValueError(
            f"understanding.json 中 pinyin 数量({len(llm_py)}) "
            f"与 text 汉字数({len(llm_chars)})不一致")

    audio_chars = [c["char"] for c in asr["chars"]]
    field = _audio_pinyin_field()
    audio_py = [c.get(field) or c["pinyin"] for c in asr["chars"]]

    # 把音频拼音序列对齐到原稿字序列 → audio_aligned[i] = (char, pinyin) 或 None
    audio_aligned: list[tuple[str, str] | None] = [None] * len(llm_py)
    for op in Levenshtein.opcodes(llm_py, audio_py):
        tag, i1, i2, j1, j2 = op.tag, op.src_start, op.src_end, op.dest_start, op.dest_end
        if tag == "equal":
            for k in range(i2 - i1):
                audio_aligned[i1 + k] = (audio_chars[j1 + k], audio_py[j1 + k])
        elif tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                audio_aligned[i1 + k] = (audio_chars[j1 + k], audio_py[j1 + k])
        # delete/insert: 原稿位置无音频对应 → 保持 None

    diffs: list[DiffItem] = []
    covered = 0
    correct = 0
    for i, ch in enumerate(llm_chars):
        a = audio_aligned[i]
        if a is None:
            diffs.append(DiffItem(
                kind="audio_missing", ref_index=i + 1, ref_char=ch,
                llm_pinyin=llm_py[i], audio_pinyin=None, audio_char=None))
            continue
        covered += 1
        a_char, a_py = a
        if pinyin_equiv(llm_py[i], a_py, a_char):
            correct += 1
        else:
            diffs.append(DiffItem(
                kind="wrong", ref_index=i + 1, ref_char=ch,
                llm_pinyin=llm_py[i], audio_pinyin=a_py, audio_char=a_char))

    acc = correct / covered if covered else 0.0
    wrongs = [d for d in diffs if d.kind == "wrong"]
    return CheckResult(
        model_label=model_label,
        llm_text=und["text"],
        audio_text=asr["text"],
        ref_han_count=len(llm_chars),
        char_accuracy=round(acc, 4),
        is_match=not wrongs,
        diffs=diffs,
    )


def format_report(res: CheckResult) -> str:
    lines = [
        f"被评估模型: {res.model_label}",
        f"原稿(汉字数): {res.ref_han_count}",
        f"音频识别文本: {res.audio_text}",
        f"拼音理解准确率(相对音频): {res.char_accuracy:.2%}",
    ]
    wrongs = [d for d in res.diffs if d.kind == "wrong"]
    missings = [d for d in res.diffs if d.kind == "audio_missing"]
    if not wrongs:
        lines.append("结论: 一致 ✔ 该模型对原稿的拼音理解与音频读音一致")
    else:
        lines.append(f"结论: 不一致 ✘ 共 {len(wrongs)} 处理解偏差:")
        for d in wrongs:
            lines.append(f"  [偏差] 第{d.ref_index}字「{d.ref_char}」 "
                         f"模型理解: {d.llm_pinyin}  音频读音: {d.audio_pinyin}"
                         f" (识别字「{d.audio_char}」)")
    if missings:
        lines.append(f"注: {len(missings)} 字音频未覆盖（Whisper 漏识别），未计入准确率")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="对比 LLM 原稿拼音理解 与 Whisper 音频读音")
    ap.add_argument("understanding", help="被评估模型产出的 understanding.json")
    ap.add_argument("asr_json", help="asr/cache/<音频>.json")
    ap.add_argument("--label", default="", help="被评估模型名称 (默认取 understanding 同目录名)")
    ap.add_argument("--strict-tone", action="store_true", help="严格比对声调")
    ap.add_argument("--out", default=str(Path(__file__).parent / "report"),
                    help="报告输出目录 (默认本目录 report/)")
    args = ap.parse_args()

    global STRICT_TONE
    STRICT_TONE = args.strict_tone

    und = json.loads(Path(args.understanding).read_text(encoding="utf-8"))
    asr = json.loads(Path(args.asr_json).read_text(encoding="utf-8"))
    label = args.label or Path(args.understanding).parent.name

    res = compare(und, asr, model_label=label)
    body = format_report(res)
    print(body)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = (
        f"# 理解评估报告 — {label}\n\n"
        f"- 时间: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- 音频: {asr.get('audio', '')}\n"
        f"- Whisper 模型: {asr.get('whisper_model', '')}\n"
        f"- 声调比对: {'严格' if STRICT_TONE else '忽略'}\n\n"
    )
    (out_dir / "report.md").write_text(header + body + "\n", encoding="utf-8")
    d = asdict(res)
    d["diffs"] = [asdict(x) for x in res.diffs]
    (out_dir / "report.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {out_dir}/report.md 与 report.json")

    return 0 if res.is_match else 2


if __name__ == "__main__":
    sys.exit(main())
