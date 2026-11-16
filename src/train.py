"""Compare models under random cross-validation and leave-one-site-out evaluation."""
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.prepare import CORE_DROP, TARGET, build_preprocessor, load_combined, split_xy

results_dir = Path("results")
seed = 1

feature_sets = {"full": (), "core": CORE_DROP}

def make_models(drop_features):
    """Return unfitted model pipelines for one feature set"""
    return {
        "logistic": Pipeline([
            ("prep", build_preprocessor(drop_features)),
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "svm": Pipeline([
            ("prep", build_preprocessor(drop_features)),
            ("model", SVC(probability=True, random_state=seed)),
        ]),
        # XGBoost needs no scaling and handles missing values natively,
        # so imputation is switched off for it.
        "xgboost": Pipeline([
            ("prep", build_preprocessor(drop_features, scale=False, impute=False)),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=seed,
            )),
        ]),
    }


def cv_evaluate(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Stratified k-fold CV on the pooled data (patients from all sites are mixed)."""
    X, y = split_xy(df)
    strata = df["site"] + "_" + df[TARGET].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    for fs_name, drop in feature_sets.items():
        for model_name, model in make_models(drop).items():
            aucs, briers = [], []
            for train_idx, test_idx in skf.split(X, strata):
                fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
                p = fitted.predict_proba(X.iloc[test_idx])[:, 1]
                aucs.append(roc_auc_score(y.iloc[test_idx], p))
                briers.append(brier_score_loss(y.iloc[test_idx], p))
            rows.append({
                "feature_set": fs_name, "model": model_name,
                "auc_mean": pd.Series(aucs).mean(), "auc_sd": pd.Series(aucs).std(),
                "brier_mean": pd.Series(briers).mean(),
            })
            print(f"CV   {fs_name:5s} {model_name:9s} AUC={rows[-1]['auc_mean']:.3f}")
    return pd.DataFrame(rows)

def loso_evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-site-out: train on three sites, test on the fourth."""
    X, y = split_xy(df)
    sites = df["site"]

    rows = []
    for fs_name, drop in feature_sets.items():
        for model_name, model in make_models(drop).items():
            for site in sorted(sites.unique()):
                test = (sites == site).to_numpy()
                fitted = clone(model).fit(X.loc[~test], y.loc[~test])
                p = fitted.predict_proba(X.loc[test])[:, 1]
                rows.append({
                    "feature_set": fs_name, "model": model_name,
                    "held_out_site": site,
                    "auc": roc_auc_score(y.loc[test], p),
                    "brier": brier_score_loss(y.loc[test], p),
                    "observed_prevalence": y.loc[test].mean(),
                    "mean_predicted": p.mean(),
                })
            print(f"LOSO {fs_name:5s} {model_name:9s} done")
    return pd.DataFrame(rows)

def main() -> None:
    df = load_combined()
    results_dir.mkdir(exist_ok=True)

    cv = cv_evaluate(df)
    loso = loso_evaluate(df)
    cv.to_csv(results_dir / "cv_results.csv", index=False)
    loso.to_csv(results_dir / "loso_results.csv", index=False)

    print("\n=== Stratified 5-fold CV (pooled sites) ===")
    print(cv.round(3).to_string(index=False))

    print("\n=== Leave-one-site-out AUC ===")
    pivot = loso.pivot_table(
        index=["feature_set", "model"], columns="held_out_site", values="auc"
    )
    pivot["mean"] = pivot.mean(axis=1)
    print(pivot.round(3))

    print("\n=== Calibration check: mean predicted risk vs observed prevalence ===")
    cal = loso[loso["model"] == "logistic"].pivot_table(
        index="feature_set", columns="held_out_site", values="mean_predicted"
    )
    print("Observed prevalence:\n", loso.groupby("held_out_site")["observed_prevalence"].first().round(3))
    print("Logistic mean predicted risk:\n", cal.round(3))


if __name__ == "__main__":
    main()
