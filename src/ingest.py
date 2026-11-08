# Download the four UCI Heart Disease site files and build one combined data set
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
sites = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
raw_dir = Path("data/raw")
processed_dir = Path("data/processed")

def download_raw() -> None:
    # Download each site file into data/raw
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in sites.values():
        path = raw_dir / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            urlretrieve(url + filename, path)

def load_site(site: str, filename: str) -> pd.DataFrame:
    """Read one site file, mark missing values, and tag the site."""
    df = pd.read_csv(raw_dir / filename, header=None, names=columns, na_values="?")
    df["site"] = site
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix impossible values and build the binary target."""
    df = df.copy()
    # A resting blood pressure or cholesterol of 0 is not physiologically possible,
    # so these zeros are really missing values.
    for col in ["trestbps", "chol"]:
        df.loc[df[col] == 0, col] = np.nan
    # Original target 'num' is 0 (no disease) to 4 (severity); binarize it.
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns="num")

def main() -> None:
    download_raw()
    combined = pd.concat(
        [load_site(site, fname) for site, fname in sites.items()],
        ignore_index=True,
    )
    combined = clean(combined)

    processed_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(processed_dir / "heart_combined.csv", index=False)

    print(f"\nRows per site:\n{combined['site'].value_counts()}")
    print(f"\nDisease prevalence by site:\n{combined.groupby('site')['target'].mean().round(3)}")
    print(f"\nMissing values (%) by site:")
    print((combined.drop(columns='target').isna().groupby(combined['site']).mean() * 100).round(1).T)


if __name__ == "__main__":
    main()