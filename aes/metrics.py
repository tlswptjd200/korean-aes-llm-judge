"""채점 성능 지표. 점수는 두 채점자 평균(1.0~5.0, 0.5 단위)이므로 x2 해서 정수 등급으로 QWK를 잰다."""
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score


def to_grade(s):
    return np.clip(np.rint(np.asarray(s, dtype=float) * 2), 2, 10).astype(int)


def qwk_ci(y_true, y_pred, n_boot=1000, seed=0):
    """QWK의 부트스트랩 95% 신뢰구간."""
    gt, gp = to_grade(y_true), to_grade(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gt), (n_boot, len(gt)))
    ks = [cohen_kappa_score(gt[i], gp[i], weights="quadratic", labels=list(range(2, 11))) for i in idx]
    return np.nanpercentile(ks, 2.5), np.nanpercentile(ks, 97.5)


def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    gt, gp = to_grade(y_true), to_grade(y_pred)
    return {
        "QWK": cohen_kappa_score(gt, gp, weights="quadratic", labels=list(range(2, 11))),
        "Pearson": pearsonr(y_true, y_pred)[0] if np.std(y_pred) > 0 else 0.0,
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "Exact": float(np.mean(gt == gp)),
        "pred_std": float(np.std(y_pred)),
    }
