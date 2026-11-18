"""Test whether recalibration fixes poor probabilities at unseen sites."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.prepare import load_combined, split_xy
from src.train import feature_sets, seed, make_models

results_dir = Path("results")
methods = ["none", "intercept_only", "logistic"]
eps = 1e-6

def to_logit(p: np.ndarray) -> np.ndarray:
    """Convert probabilities to log-odds."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def fit_intercept_only(logit_p: np.ndarray, y: np.ndarray) -> float:
    """Find the single shift b that minimizes log loss of sigmoid(logit_p + b)."""
    def nll(b):
        z = logit_p + b
        return np.mean(np.logaddexp(0, z) - y * z)
    return minimize_scalar(nll, bounds=(-10, 10), method="bounded").x

def recalibrate(method, lp_train, y_train, lp_test):
    """Fit a recalibrator on (lp_train, y_train) and apply it to lp_test."""
    if method == "none":
        return expit(lp_test)
    if method == "intercept_only":
        return expit(lp_test + fit_intercept_only(lp_train, y_train))
    # logistic: p' = sigmoid(a * logit(p) + b), essentially unpenalized
    lr = LogisticRegression(C=1e6).fit(lp_train.reshape(-1, 1), y_train)
    return lr.predict_proba(lp_test.reshape(-1, 1))[:, 1]

def crossfit(p, y, method, n_splits=5):
    """Out-of-fold recalibrated predictions within one site."""
    lp = to_logit(p)
    out = np.empty_like(lp)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(lp, y):
        out[test_idx] = recalibrate(method, lp[train_idx], y[train_idx], lp[test_idx])
    return out

def calibration_points(q, y, n_bins=5):
    """Mean predicted vs observed rate in quantile bins of predicted risk."""
    d = pd.DataFrame({"q": q, "y": y})
    d["bin"] = pd.qcut(d["q"], n_bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(pred=("q", "mean"), obs=("y", "mean"))

def plot_curves(preds, feature_set, model_name, path):
    """Calibration curves per held-out site, before and after recalibration."""
    sites = sorted({k[2] for k in preds})
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, site in zip(axes.ravel(), sites):
        for method in methods:
            q, y = preds[(feature_set, model_name, site, method)]
            pts = calibration_points(q, y)
            ax.plot(pts["pred"], pts["obs"], marker="o", label=method)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"Held-out site: {site}")
        ax.set_xlabel("Mean predicted risk"); ax.set_ylabel("Observed fraction with disease")
    axes.ravel()[0].legend()
    fig.suptitle(f"Calibration curves ({model_name}, {feature_set} features)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main() -> None:
    df = load_combined()
    X, y = split_xy(df)
    sites = df["site"]
    results_dir.mkdir(exist_ok=True)

    rows, preds = [], {}
    for fs_name, drop in feature_sets.items():
        for model_name, model in make_models(drop).items():
            for site in sorted(sites.unique()):
                test = (sites == site).to_numpy()
                fitted = clone(model).fit(X.loc[~test], y.loc[~test])
                p = fitted.predict_proba(X.loc[test])[:, 1]
                y_te = y.loc[test].to_numpy()

                for method in methods:
                    q = crossfit(p, y_te, method)
                    q = np.clip(q, eps, 1 - eps)
                    preds[(fs_name, model_name, site, method)] = (q, y_te)
                    rows.append({
                        "feature_set": fs_name, "model": model_name,
                        "held_out_site": site, "method": method,
                        "auc": roc_auc_score(y_te, q),
                        "brier": brier_score_loss(y_te, q),
                        "log_loss": log_loss(y_te, q, labels=[0, 1]),
                        "mean_predicted": q.mean(),
                        "observed_prevalence": y_te.mean(),
                    })
            print(f"{fs_name:5s} {model_name:9s} done")

    res = pd.DataFrame(rows)
    res.to_csv(results_dir/ "calibration_results.csv", index=False)

    idx = ["feature_set", "model", "method"]
    print("\n=== Brier score by held-out site ===")
    print(res.pivot_table(index=idx, columns="held_out_site", values="brier").round(3))

    print("\n=== Log loss by held-out site  ===")
    print(res.pivot_table(index=idx, columns="held_out_site", values="log_loss").round(3))

    print("\n=== Mean predicted risk by held-out site ===")
    print(res.pivot_table(index=idx, columns="held_out_site", values="mean_predicted").round(3))

    prev = res.groupby("held_out_site")["observed_prevalence"].first()
    print("\nObserved prevalence:\n", prev.round(3))
    print("\nReference Brier score of always predicting the site's own prevalence:\n",
          (prev * (1 - prev)).round(3))

    plot_curves(preds, "core", "logistic", results_dir / "calibration_curves.png")
    print("\nSaved results/calibration_results.csv and results/calibration_curves.png")


if __name__ == "__main__":
    main()