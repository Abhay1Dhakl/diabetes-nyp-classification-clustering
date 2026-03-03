from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_MODEL = Path("models/diabetes_pipeline.joblib")


def load_input(path: Path) -> pd.DataFrame:
    """Load and sanitize input features for prediction.

    Args:
        path: Path to the input CSV file.
    Returns:
        Dataframe of cleaned feature rows.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "CLASS" in df.columns:
        df = df.drop(columns=["CLASS"])

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()

    id_cols = [c for c in ["ID", "No_Pation"] if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)

    return df


def main() -> None:
    """Run batch predictions from the CLI.

    Args:
        None.
    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Run predictions using the saved pipeline.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to joblib model")
    parser.add_argument("--input", type=Path, required=True, help="CSV file with feature rows")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    model = joblib.load(args.model)
    X = load_input(args.input)

    preds = model.predict(X)

    output_df = X.copy()
    output_df["prediction"] = preds

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        for idx, label in enumerate(model.classes_):
            output_df[f"proba_{label}"] = proba[:, idx]

    if args.output is None:
        print(output_df.head(20).to_string(index=False))
    else:
        output_df.to_csv(args.output, index=False)
        print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
