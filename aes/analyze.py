"""모든 실험 결과를 모아 표(out/summary.csv)와 그림(figs/*.png)을 만든다."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from baselines import make_models, fit_predict
from llm_judge import CACHE, INJECTION
from load_data import OUT
from metrics import evaluate, qwk_ci

FIG = OUT.parent / "figs"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

COND_NAME = {"zs": "Zero-shot", "fs": "Few-shot(3)", "zs_inj": "Zero-shot+주입",
             "fs_inj": "Few-shot+주입", "zs_def_inj": "방어프롬프트+주입"}


def load_llm():
    recs = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines()]
    d = pd.DataFrame(recs).drop_duplicates(["cond", "file"], keep="last")
    return d[d.pred.notna()]


def main():
    FIG.mkdir(exist_ok=True)
    df = pd.read_json(OUT / "essays.jsonl", lines=True)
    tr, va = df[df.split == "train"], df[df.split == "valid"]
    llm = load_llm()
    sample_files = set(pd.read_csv(OUT / "llm_sample.csv").file)
    sub = va[va.file.isin(sample_files)].reset_index(drop=True)

    # 1) 전체 valid 2000편에서의 모델 성능 (베이스라인 + 임베딩)
    full = pd.read_csv(OUT / "baselines.csv")
    if (OUT / "embed.csv").exists():
        full = pd.concat([full, pd.read_csv(OUT / "embed.csv")])
    full.to_csv(OUT / "summary_full.csv", index=False, encoding="utf-8-sig")

    # 2) LLM 샘플(149편)에서 같은 조건으로 비교
    bp = pd.read_csv(OUT / "baseline_preds.csv").set_index("file")
    ep = pd.read_csv(OUT / "embed_preds.csv").set_index("file") if (OUT / "embed_preds.csv").exists() else None
    rows, _pairs = [], []

    def add(name, y, p):
        y, p = np.asarray(y, float), np.asarray(p, float)
        rows.append({"model": name, "n": len(y), **evaluate(y, p)})
        _pairs.append((y, p))

    r = sub.holistic_raters
    add("사람: 채점자1 vs 채점자2", r.str[0], r.str[1])
    for col in ["TF-IDF char2-4 + Ridge", "TF-IDF char2-4 + Ridge [cross]"]:
        add(col, sub.holistic, bp.loc[sub.file, col])
    if ep is not None:
        for col in ep.columns.drop("y"):
            add(col, sub.holistic, ep.loc[sub.file, col])
    piv = llm.pivot(index="file", columns="cond", values="pred")
    for c in COND_NAME:
        if c in piv:
            s = piv[c].reindex(sub.file)
            m = s.notna().values
            add(f"Gemini {COND_NAME[c]}", sub.holistic[m], s[m])
    res = pd.DataFrame(rows)
    res["pred_mean"] = np.nan
    for i, name in enumerate(res.model):
        y, p = _pairs[i]
        lo, hi = qwk_ci(y, p)
        res.loc[i, ["QWK_lo", "QWK_hi", "pred_mean"]] = lo, hi, float(np.mean(p))
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT / "summary_sample.csv", index=False, encoding="utf-8-sig")

    # 3) 주입 공격 효과: 같은 글에서 점수 변화
    model = make_models()["TF-IDF char2-4 + Ridge"]
    clean = fit_predict("TF-IDF", model, tr, sub)
    inj = np.clip(model.predict(sub.text + INJECTION), 1, 5)
    att = [{"채점기": "TF-IDF+Ridge", "평균 변화": float(np.mean(inj - clean)),
            "상승 비율": float(np.mean(inj - clean > 0.05)), "5점 비율": float(np.mean(inj >= 4.75))}]
    pairs = [("zs", "zs_inj"), ("fs", "fs_inj"), ("zs", "zs_def_inj")]
    for a, b in pairs:
        if a in piv and b in piv:
            p = piv[[a, b]].dropna()
            delta = p[b] - p[a]
            stat = wilcoxon(p[b], p[a], zero_method="zsplit").pvalue if (delta != 0).any() else 1.0
            att.append({"채점기": f"Gemini {COND_NAME[a]} → {COND_NAME[b]}", "n": len(p),
                        "평균 변화": float(delta.mean()), "상승 비율": float((delta > 0).mean()),
                        "5점 비율": float((p[b] == 5).mean()), "Wilcoxon p": stat})
    att = pd.DataFrame(att)
    print(att.round(3).to_string(index=False))
    att.to_csv(OUT / "attack.csv", index=False, encoding="utf-8-sig")

    # --- 그림 ---
    # (a) in-prompt vs cross-prompt
    f = full[full.setting.isin(["in-prompt", "cross-prompt"]) & ~full.model.str.startswith("Constant")]
    pv = f.pivot(index="model", columns="setting", values="QWK")[["in-prompt", "cross-prompt"]]
    ax = pv.plot.bar(figsize=(6.5, 3.8), rot=0, color=["#4C78A8", "#F58518"])
    human = full.loc[full.setting == "human", "QWK"].iloc[0]
    ax.axhline(human, ls="--", c="gray")
    ax.text(-0.4, human + 0.01, f"사람 간 일치 {human:.2f}", color="gray", fontsize=9)
    ax.set_ylabel("QWK (valid 2,000편)"); ax.set_xlabel("")
    ax.set_title("처음 보는 문항(cross-prompt)에서의 성능 하락")
    ax.set_xticklabels([t.get_text().replace(" + ", "\n+ ") for t in ax.get_xticklabels()], fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "fig1_cross_prompt.png", dpi=200); plt.close()

    # (b) 샘플에서의 QWK 비교
    rs = res[~res.model.str.contains("주입")]
    plt.figure(figsize=(7, 3.8))
    plt.barh(rs.model, rs.QWK, color="#54A24B"); plt.gca().invert_yaxis()
    plt.xlabel("QWK (LLM 샘플 149편)"); plt.title("채점기별 사람 점수와의 일치도")
    plt.tight_layout(); plt.savefig(FIG / "fig2_qwk_sample.png", dpi=200); plt.close()

    # (c) 점수 분포: 사람 vs LLM
    conds = [c for c in ["zs", "fs", "zs_inj", "zs_def_inj"] if c in piv]
    fig, axes = plt.subplots(1, len(conds) + 1, figsize=(3 * (len(conds) + 1), 3), sharey=True)
    axes[0].hist(sub.holistic, bins=np.arange(0.75, 5.5, 0.5), color="gray"); axes[0].set_title("사람(평균)")
    for ax, c in zip(axes[1:], conds):
        ax.hist(piv[c].dropna(), bins=np.arange(0.5, 5.6, 1), color="#E45756", rwidth=0.8)
        ax.set_title(COND_NAME[c])
    for ax in axes:
        ax.set_xticks(range(1, 6))
    plt.tight_layout(); plt.savefig(FIG / "fig3_score_dist.png", dpi=200); plt.close()

    # (d) 혼동 행렬 (zs, fs)
    for c in [c for c in ["zs", "fs"] if c in piv]:
        p = piv[c].reindex(sub.file)
        m = p.notna().values
        ct = pd.crosstab(np.rint(sub.holistic[m].values).astype(int), p[m].values.astype(int),
                         rownames=["사람(반올림)"], colnames=["Gemini"])
        ct.to_csv(OUT / f"confusion_{c}.csv", encoding="utf-8-sig")
        print(c, "\n", ct)


if __name__ == "__main__":
    main()
