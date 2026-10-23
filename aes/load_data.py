"""AI Hub '논술형 글쓰기 평가 데이터' 라벨링 zip(TL_*, VL_*)을 읽어 하나의 표로 저장한다.

출력: aes/out/essays.jsonl  (1행 = 1편의 글)
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "26.논술형 글쓰기 평가 데이터" / "3.개방데이터" / "1.데이터"
OUT = ROOT / "out"

TRAITS = ["task_1", "content_1", "content_2", "content_3",
          "organization_1", "organization_2", "expression_1", "expression_2"]


def decode(raw: bytes) -> dict:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError("decode failed")


def parse(d: dict, split: str) -> dict:
    q, a = d["essay_question"], d["essay_answer"]
    hol = d["score"]["personal"]["holistic"]
    ana = d["score"]["personal"]["analytic"]
    row = {
        "split": split,
        "essay_id": a["id"],
        "q_id": q["id"],
        "subject": q["subject"],
        "grade": q["grade"],
        "type": q["type"],
        "topic": q["topic"],
        "prompt": q["prompt"],
        "text": a["text"],
        "len_syllable": a["len_syllable"],
        "len_word": a["len_word"],
        "holistic_raters": hol["score"],
        "holistic": sum(hol["score"]) / len(hol["score"]),
        "holistic_feedback": hol.get("feedback", ""),
        "rubric": d["rubric"],
    }
    for t in TRAITS:
        if t in ana:
            row[t] = sum(ana[t]["score"]) / len(ana[t]["score"])
    return row


def main():
    OUT.mkdir(exist_ok=True)
    rows, errors = [], 0
    for split, sub in (("train", "Training"), ("valid", "Validation")):
        for zp in sorted((DATA / sub / "02.라벨링데이터").glob("*.zip")):
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if not name.endswith(".json"):
                        continue
                    try:
                        r = parse(decode(z.read(name)), split)
                        r["file"] = name.strip("/")
                        rows.append(r)
                    except Exception as e:  # 손상/구조 다른 파일은 건너뛰고 개수만 보고
                        errors += 1
                        print(f"skip {zp.name}/{name}: {e!r}", file=sys.stderr)
    with open(OUT / "essays.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved {len(rows)} essays ({errors} skipped) -> {OUT / 'essays.jsonl'}")


if __name__ == "__main__":
    main()
