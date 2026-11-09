"""Download, load and clean the UCI Diabetes 130-US Hospitals dataset."""
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

log = logging.getLogger(__name__)

TARGET = "readmitted_30d"
GROUP = "patient_nbr"

# Discharge dispositions meaning the patient died or went to hospice:
# they cannot be readmitted, so keeping them would teach the model nonsense.
EXCLUDED_DISCHARGE = {11, 13, 14, 19, 20, 21}


def download(url: str, raw_dir: Path, raw_file: str) -> Path:
    target = Path(raw_dir) / raw_file
    if target.exists():
        return target
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset from %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                (Path(raw_dir) / Path(name).name).write_bytes(zf.read(name))
    return target


def load_raw(path: Path) -> pd.DataFrame:
    # The dataset encodes missing values as "?".
    return pd.read_csv(path, na_values="?", low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE)]
    df = df[df["gender"] != "Unknown/Invalid"]
    df[TARGET] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])
    log.info("Cleaned %d -> %d rows (positive rate %.3f)", n0, len(df), df[TARGET].mean())
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float, valid_size: float, random_state: int):
    """Split by patient so the same person never appears in train and test.

    Many patients have several encounters; a row-level split would leak
    patient identity and inflate every metric.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trval_idx, test_idx = next(gss.split(df, groups=df[GROUP]))
    trval, test = df.iloc[trval_idx], df.iloc[test_idx]

    gss = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=random_state)
    tr_idx, va_idx = next(gss.split(trval, groups=trval[GROUP]))
    train, valid = trval.iloc[tr_idx], trval.iloc[va_idx]

    assert not set(train[GROUP]) & set(test[GROUP])
    assert not set(valid[GROUP]) & set(test[GROUP])
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET].to_numpy(dtype=np.int64)
