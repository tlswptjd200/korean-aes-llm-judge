"""한국어 문장 임베딩(SBERT) + Ridge 회귀. GPU 없이 돌도록 train은 문항별 층화 샘플링한다.

모델: jhgan/ko-sroberta-multitask (KLUE-RoBERTa-base 기반, max_seq_length=256)
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV

from baselines import prompt_folds, SEED
from load_data import OUT
from metrics import evaluate

MODEL = "jhgan/ko-sroberta-multitask"
PER_PROMPT = 20  # 문항당 train 샘플 수 (89문항 x 20 = 1,780편). CPU 시간 제약


def embed(model, texts, path):
    if path.exists():
        return np.load(path)
    E = model.encode(list(texts), batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    np.save(path, E)
    return E


def main():
    df = pd.read_json(OUT / "essays.jsonl", lines=True)
    tr = (df[df.split == "train"].sample(frac=1, random_state=SEED)
          .groupby("q_id").head(PER_PROMPT).reset_index(drop=True))
    va = df[df.split == "valid"].reset_index(drop=True)
    print(f"train sample {len(tr)}, valid {len(va)}")

    model = SentenceTransformer(MODEL, device="cpu")
    model.max_seq_length = 256  # CPU 시간 제약으로 앞 256토큰만 사용
    E_tr = embed(model, tr.text, OUT / "emb_train.npy")
    E_va = embed(model, va.text, OUT / "emb_valid.npy")

    alphas = np.logspace(-2, 3, 12)
    rows, preds = [], {}
    m = RidgeCV(alphas=alphas).fit(E_tr, tr.holistic)
    preds["SBERT+Ridge"] = np.clip(m.predict(E_va), 1, 5)
    rows.append({"setting": "in-prompt", "model": "SBERT+Ridge", **evaluate(va.holistic, preds["SBERT+Ridge"])})

    fold_of = prompt_folds(df)
    p = np.zeros(len(va))
    for k in range(5):
        trm = (tr.q_id.map(fold_of) != k).values
        tem = (va.q_id.map(fold_of) == k).values
        p[tem] = np.clip(RidgeCV(alphas=alphas).fit(E_tr[trm], tr.holistic[trm]).predict(E_va[tem]), 1, 5)
    preds["SBERT+Ridge [cross]"] = p
    rows.append({"setting": "cross-prompt", "model": "SBERT+Ridge", **evaluate(va.holistic, p)})

    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT / "embed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"file": va.file, "y": va.holistic, **preds}).to_csv(OUT / "embed_preds.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
