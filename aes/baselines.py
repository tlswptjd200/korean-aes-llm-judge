"""전통적 AES 베이스라인: 상수, 길이, TF-IDF(char n-gram)+Ridge.

두 가지 설정으로 평가한다.
  in-prompt   : train 전체로 학습 → valid 전체 평가 (같은 문항이 학습에 있음)
  cross-prompt: 문항(q_id)을 5개 그룹으로 나눠, 4개 그룹의 train으로 학습 → 남은 그룹의 valid 평가
                (수업에서 말한 "주제가 바뀌면 점수가 무너진다"를 재현)
"""
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import evaluate, to_grade, cohen_kappa_score
from load_data import OUT

SEED = 0


def length_feats(df):
    t = df.text
    return np.c_[df.len_syllable, df.len_word, t.str.count(r"\n") + 1,
                 t.str.count(r"[.?!]"), t.apply(lambda s: len(set(s.split())) / max(1, len(s.split())))]


def prompt_folds(df, k=5):
    """q_id -> fold 번호. 모든 실험이 같은 문항 분할을 쓰도록 고정."""
    q_ids = np.array(sorted(df.q_id.unique()))
    np.random.default_rng(SEED).shuffle(q_ids)
    return {q: i % k for i, q in enumerate(q_ids)}


def make_models():
    return {
        "Constant(train mean)": None,
        "Length/surface feats + Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "TF-IDF char2-4 + Ridge": make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=200_000, sublinear_tf=True),
            Ridge(alpha=3.0)),
    }


def fit_predict(name, model, tr, te):
    if model is None:
        return np.full(len(te), tr.holistic.mean())
    X_tr, X_te = (length_feats(tr), length_feats(te)) if name.startswith("Length") else (tr.text, te.text)
    model.fit(X_tr, tr.holistic)
    return np.clip(model.predict(X_te), 1, 5)


def main():
    df = pd.read_json(OUT / "essays.jsonl", lines=True)
    tr, va = df[df.split == "train"], df[df.split == "valid"].reset_index(drop=True)
    rows = []

    # 사람 간 일치도 (상한선 참고): 채점자1 vs 채점자2
    r1 = va.holistic_raters.str[0]
    r2 = va.holistic_raters.str[1]
    rows.append({"setting": "human", "model": "Rater1 vs Rater2",
                 "QWK": cohen_kappa_score(r1, r2, weights="quadratic"),
                 "Exact": float(np.mean(r1 == r2))})

    preds = {}
    for name, model in make_models().items():
        p = fit_predict(name, model, tr, va)
        preds[name] = p
        rows.append({"setting": "in-prompt", "model": name, **evaluate(va.holistic, p)})

    # cross-prompt
    fold_of = prompt_folds(df)
    for name in make_models():
        p = np.zeros(len(va))
        for k in range(5):
            model = make_models()[name]
            tr_k = tr[tr.q_id.map(fold_of) != k]
            te_mask = (va.q_id.map(fold_of) == k).values
            p[te_mask] = fit_predict(name, model, tr_k, va[te_mask])
        preds[name + " [cross]"] = p
        rows.append({"setting": "cross-prompt", "model": name, **evaluate(va.holistic, p)})

    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT / "baselines.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"essay_id": va.essay_id, "file": va.file, "y": va.holistic, **preds}).to_csv(
        OUT / "baseline_preds.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
