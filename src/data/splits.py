from pathlib import Path
import re
import numpy as np
import pandas as pd

NAME_RE = re.compile(
    r"^(?P<identity>[fm]\d+)_(?P<session>\d+)_(?P<prompt>[a-z]\d+)"
    r"_i(?P<inst>\d+)_(?P<take>\d+)$"
)

_FOLDERS = {
    "target_train":     1,
    "target_dev":       1,
    "non_target_train": 0,
    "non_target_dev":   0,
}


def load_manifest(data_dir: Path) -> pd.DataFrame:
    rows = []
    for folder, label in _FOLDERS.items():
        for p in sorted((data_dir / folder).glob("*.png")):
            m = NAME_RE.match(p.stem)
            if not m:
                raise ValueError(f"Unexpected filename: {p.name}")
            rows.append({**m.groupdict(), "label": label, "stem": p.stem})

    df = pd.DataFrame(rows)
    df["session_id"] = df["identity"] + "_" + df["session"]
    df["group"] = np.where(df["label"] == 1, df["session_id"], df["identity"])
    return df


def _assign_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    fold_of_group: dict[str, int] = {}

    for label in [0, 1]:
        groups = list(df.loc[df["label"] == label, "group"].unique())
        rng.shuffle(groups)
        for i, g in enumerate(groups):
            fold_of_group[g] = i % n_splits

    return df["group"].map(fold_of_group)


def iter_folds(df: pd.DataFrame, n_splits: int = 3, seed: int = 0):
    """Yield (fold_id, train_idx, val_idx) for each fold.

    Groups (target by session, non-target by identity) are never split
    across train and val.
    """
    folds = _assign_folds(df, n_splits=n_splits, seed=seed)

    for fold_id in range(n_splits):
        val_idx = df.index[folds == fold_id].to_numpy()
        train_idx = df.index[folds != fold_id].to_numpy()
        yield fold_id, train_idx, val_idx


def iter_folds_loso(df: pd.DataFrame, seed: int = 0):
    """Leave-One-Session-Out on the target person.

    Number of folds equals number of unique target sessions (3 for m431).
    Non-target speakers are split into matching groups round-robin.
    """
    rng = np.random.default_rng(seed)

    target_sessions = sorted(df.loc[df["label"] == 1, "session_id"].unique())
    n_splits = len(target_sessions)

    nt_speakers = list(df.loc[df["label"] == 0, "identity"].unique())
    rng.shuffle(nt_speakers)
    nt_fold = {spk: i % n_splits for i, spk in enumerate(nt_speakers)}

    for fold_id, session in enumerate(target_sessions):
        val_mask = (
            ((df["label"] == 1) & (df["session_id"] == session)) |
            ((df["label"] == 0) & (df["identity"].map(nt_fold) == fold_id))
        )
        val_idx = df.index[val_mask].to_numpy()
        train_idx = df.index[~val_mask].to_numpy()
        yield fold_id, train_idx, val_idx
