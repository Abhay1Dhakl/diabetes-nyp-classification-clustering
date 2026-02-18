# diabetes_prediction

## Notebooks
- `notebooks/eda.ipynb`: Exploratory data analysis on the raw dataset, checks data quality, and saves a cleaned copy.
- `notebooks/preprocessing.ipynb`: Creates the train/test split and scaled arrays used by modeling notebooks.
- `notebooks/classification.ipynb`: Trains baseline classifiers and compares model performance.
- `notebooks/evaluation_error_analysis.ipynb`: Evaluates a baseline model and inspects error patterns using the **preprocessed split** (no re-preprocessing).
- `notebooks/clustering.ipynb`: Runs unsupervised clustering on the cleaned dataset and saves cluster assignments.

## Data Artifacts
- `data/raw/diabetes.csv`: Raw dataset.
- `data/processed/diabetes_clean.csv`: Cleaned dataset saved from EDA.
- `data/processed/split_scaled.npz`: Preprocessed train/test split with scaled features.
- `data/processed/diabetes_clusters.csv`: Cluster assignments created by the clustering notebook.

## Recommended Run Order
1. `notebooks/eda.ipynb`
2. `notebooks/preprocessing.ipynb`
3. `notebooks/classification.ipynb`
4. `notebooks/evaluation_error_analysis.ipynb`
5. `notebooks/clustering.ipynb`
