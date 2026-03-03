from __future__ import annotations

import numpy as np
import pandas as pd


def load_data(path) -> pd.DataFrame:
    """Load and sanitize the diabetes dataset.

    Args:
        path: Path to the cleaned CSV file.
    Returns:
        Cleaned dataframe with standardized columns.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "CLASS" not in df.columns:
        raise ValueError("Target column 'CLASS' not found in dataset.")

    df["CLASS"] = df["CLASS"].astype(str).str.strip()

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()

    id_cols = [c for c in ["ID", "No_Pation"] if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)

    assert_no_missing(df)
    return df


def assert_no_missing(df: pd.DataFrame) -> None:
    """Validate that the dataframe has no missing or invalid values.

    Args:
        df: Input dataframe to validate.
    Returns:
        None.
    """
    na_counts = df.isna().sum()

    empty_counts = pd.Series(0, index=df.columns, dtype="int64")
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        empty_counts.loc[obj_cols] = (
            df[obj_cols].astype(str).apply(lambda s: s.str.strip().eq("")).sum()
        )

    inf_counts = pd.Series(0, index=df.columns, dtype="int64")
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        inf_counts.loc[num_cols] = df[num_cols].isin([np.inf, -np.inf]).sum()

    issue_counts = na_counts.add(empty_counts, fill_value=0).add(inf_counts, fill_value=0)
    if issue_counts.sum() > 0:
        bad = issue_counts[issue_counts > 0].sort_values(ascending=False)
        details = ", ".join(f"{col}={int(cnt)}" for col, cnt in bad.items())
        raise ValueError(
            "Missing/invalid values detected. "
            f"Columns with issues: {details}. "
            "Clean the data or re-enable imputing."
        )
