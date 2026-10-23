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
        f"SBERT 채점기는 in-prompt에서 {rep['SB_IN']}로 TF-IDF보다 낮았고, cross-prompt에서는 {rep['SB_CR']}로 약 {drop:.0f}% 감소하였다. "
        "사용한 `ko-sroberta-multitask`는 NLI·STS(문장 의미 유사도) 과제로 학습된 모델이므로, 임베딩이 글의 완성도보다 "
        "주제 정보를 주로 반영했을 가능성이 있다. 이 경우 문항이 바뀌면 임베딩 분포 자체가 달라져 선형 회귀의 성능이 떨어진다. "
        "또한 앞 256토큰만 사용하고 학습 데이터를 1,780편으로 줄인 점도 성능에 영향을 주었을 것이다. "
        "사전학습 인코더를 채점에 활용하려면 고정 임베딩보다는 채점 과제에 맞춘 파인튜닝이 필요할 것으로 보인다.")

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
