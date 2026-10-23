"""report.md의 {{...}} 자리표시자를 aes/out/*.csv 결과로 채워 report_filled.md를 만든다."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "aes" / "out"


def f3(x):
    return f"{x:.3f}"


def main(repo_url="(GitHub 저장소 링크)"):
    md = (HERE / "report.md").read_text(encoding="utf-8")
    full = pd.read_csv(OUT / "summary_full.csv")
    sb = full[full.model == "SBERT+Ridge"].set_index("setting")
    rep = {
        "SB_IN": f3(sb.loc["in-prompt", "QWK"]), "SB_CR": f3(sb.loc["cross-prompt", "QWK"]),
        "SB_IN_P": f3(sb.loc["in-prompt", "Pearson"]), "SB_CR_P": f3(sb.loc["cross-prompt", "Pearson"]),
        "REPO_URL": repo_url,
    }
    drop = (1 - sb.loc["cross-prompt", "QWK"] / sb.loc["in-prompt", "QWK"]) * 100
    rep["SB_COMMENT"] = (
        f"SBERT 임베딩은 의미 수준의 표현이라 문항이 바뀌어도 덜 흔들릴 것으로 예상했다. 그러나 결과는 반대였다. "
        f"in-prompt에서도 TF-IDF보다 낮았고({rep['SB_IN']}), cross-prompt에서는 {rep['SB_CR']}로 {drop:.0f}% 하락했다. "
        "`ko-sroberta-multitask`는 NLI·STS(문장 의미 유사도)로 학습된 모델이라 임베딩 공간이 \"무엇에 대한 글인가\"(주제)로 "
        "주로 조직되고, \"얼마나 잘 쓴 글인가\"(질)에 대한 정보는 상대적으로 약하다. 문항이 바뀌면 임베딩이 주제 축을 따라 통째로 "
        "이동하므로, 고정된 임베딩 위의 선형 회귀는 질 신호를 찾지 못한다. 앞 256토큰만 쓴 점과 train을 1,780편으로 줄인 점도 "
        "불리하게 작용했다. 사전학습 표현을 채점에 쓰려면 고정 임베딩이 아니라 채점 목표로 파인튜닝해야 함을 보여 준다.")

    s = pd.read_csv(OUT / "summary_sample.csv")
    rows = []
    for r in s.itertuples():
        mean = "–" if r.model.startswith("사람") else f"{r.pred_mean:.2f}"
        rows.append(f"| {r.model} | {r.QWK:.3f} [{r.QWK_lo:.2f}, {r.QWK_hi:.2f}] | {r.Pearson:.3f} | {r.RMSE:.3f} | {mean} |")
    rep["SAMPLE_TABLE"] = "\n".join(rows)

    a = pd.read_csv(OUT / "attack.csv")
    rows = []
    for r in a.itertuples():
        p = "–" if pd.isna(r[6]) else ("<0.001" if r[6] < 0.001 else f"{r[6]:.3f}")
        rows.append(f"| {r[1]} | {r[2]:+.2f} | {r[3]*100:.0f}% | {r[4]*100:.0f}% | {p} |")
    rep["ATTACK_TABLE"] = "\n".join(rows)

    ct = pd.read_csv(OUT / "confusion_zs.csv", index_col=0)
    lines = ["| 사람 \\ Gemini | " + " | ".join(ct.columns) + " |", "|" + "---|" * (len(ct.columns) + 1)]
    lines += [f"| {i} | " + " | ".join(str(v) for v in row) + " |" for i, row in ct.iterrows()]
    rep["CONFUSION_ZS"] = "\n".join(lines)

    for k, v in rep.items():
        md = md.replace("{{" + k + "}}", v)
    assert "{{" not in md, md[md.index("{{"):md.index("{{") + 40]
    (HERE / "report_filled.md").write_text(md, encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main(*sys.argv[1:])
