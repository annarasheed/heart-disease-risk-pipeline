"""Prepare the combined heart disease data for modelling.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data_path = Path("data/processed/heart_combined.csv")

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
NOMINAL = ["cp", "restecg", "slope", "thal"]# unordered categories
TARGET = "target"
CORE_DROP = ("ca", "thal", "slope")  # mostly missing outside Cleveland
def load_combined(path: Path = data_path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_cleveland_complete_cases(path: Path = data_path) -> pd.DataFrame:
    """Cleveland only dataset, rows with missing values dropped (297 rows)"""
    df = load_combined(path)
    return df[df["site"] == "cleveland"].dropna().reset_index(drop=True)

def split_data(df: pd.DataFrame, test_size: float = 0.30, random_state: int = 1):
    """Stratified train/test split, balanced on both site and target"""
    strata = df["site"] + "_" + df[TARGET].astype(str)
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strata
    )

def split_xy(df: pd.DataFrame):
    """Separate features from the target (the site column is kept out of X)"""
    return df.drop(columns=[TARGET, "site"]), df[TARGET]

def build_preprocessor(drop_features=(), scale=True, impute=True) -> ColumnTransformer:
    """Preprocessing step to place at the front of a model Pipeline.

    scale=False and impute=False suit XGBoost, which handles unscaled
    features and missing values natively.
    """
    drop = set(drop_features)
    numeric = [c for c in NUMERIC if c not in drop]
    binary = [c for c in BINARY if c not in drop]
    nominal = [c for c in NOMINAL if c not in drop]

    num_steps = []
    if impute:
        num_steps.append(("impute", SimpleImputer(strategy="median")))
    if scale:
        num_steps.append(("scale", StandardScaler()))

    cat_steps = []
    if impute:
        cat_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))

    bin_pipe = (
        Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
        if impute else "passthrough"
    )

    return ColumnTransformer([
        ("num", Pipeline(num_steps) if num_steps else "passthrough", numeric),
        ("bin", bin_pipe, binary),
        ("cat", Pipeline(cat_steps), nominal),
    ])

if __name__ == "__main__":
    cle = load_cleveland_complete_cases()
    print("Cleveland complete cases:", cle.shape)  # expect (297, 15)

    df = load_combined()
    train, test = split_data(df)
    print("Train/test:", train.shape, test.shape)
    print(train.groupby("site")[TARGET].agg(["count", "mean"]).round(3))

    X_train, y_train = split_xy(train)
    Xt = build_preprocessor().fit_transform(X_train)
    print("Transformed training matrix:", Xt.shape)
