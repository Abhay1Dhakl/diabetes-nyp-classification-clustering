# diabetes_prediction

## Notebooks
- `notebooks/01_eda.ipynb`: Exploratory data analysis on the raw dataset, checks data quality, and saves a cleaned copy.
- `notebooks/02_preprocessing.ipynb`: Creates the train/test split and scaled arrays used by modeling notebooks.
- `notebooks/03_classification.ipynb`: Cross-validated model comparison on training data, then test-set evaluation.
- `notebooks/04_evaluation_error_analysis.ipynb`: Evaluates a baseline model and inspects error patterns using the **preprocessed split** (no re-preprocessing).
- `notebooks/05_clustering.ipynb`: Runs unsupervised clustering on the cleaned dataset and saves cluster assignments.

## Scripts (Production-Style)
- `src/train.py`: End-to-end training with preprocessing pipeline, cross-validation, and model export.
- `src/predict.py`: Batch predictions from a CSV using the saved model.

## Hyperparameter Tuning
Training now performs hyper-parameter tuning by default using cross-validated macro‑F1.
- Search types: GridSearch for Logistic Regression and KNN, RandomizedSearch for SVM (RBF) and RandomForest.
- Scoring: `f1_macro` (handles class imbalance).
- Disable tuning if you want faster runs: use `--no-tune`.

## Parameter Rationale (Baseline Choices)
These are baseline, not guaranteed “best” parameters. They were chosen for stability and imbalance handling:
- `class_weight="balanced"`: adjusts for the skewed class distribution.
- `max_iter=3000` (LogReg): ensures convergence on scaled data.
- `C=1.0` and `kernel="rbf"` (SVM): standard non‑linear baseline.
- `n_neighbors=7` (KNN): moderate smoothing, avoids tie‑heavy small K.
- `n_estimators=300` (RandomForest): stability vs. training time trade‑off.
- `random_state=42`: reproducible splits and model fits.

## Data Artifacts
- `data/raw/diabetes.csv`: Raw dataset.
- `data/processed/diabetes_clean.csv`: Cleaned dataset saved from EDA.
- `data/processed/split_scaled.npz`: Preprocessed train/test split with scaled features.
- `data/processed/diabetes_clusters.csv`: Cluster assignments created by the clustering notebook.

## Reports
- `reports/REPORT_TEMPLATE.md`: PDF report outline you can expand and export.

## Recommended Run Order
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_classification.ipynb`
4. `notebooks/04_evaluation_error_analysis.ipynb`
5. `notebooks/05_clustering.ipynb`

## Quick Start (CLI)
```bash
python -m pip install -r requirements.txt
python src/train.py
python src/train.py --no-tune
python src/predict.py --input data/processed/diabetes_clean.csv
```
