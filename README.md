# Digicow Churn Prediction — Zindi Africa 🐄📈

**Purpose**
- Predict farmer churn/adoption targets for the Digicow Zindi competition using tabular + text features and ensemble models (CatBoost, LightGBM, stacking).

## 🔧 Quick start
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn catboost lightgbm sentence-transformers optuna
```

3. Run the main pipeline (preprocessing, features, training, submissions):

```bash
python3 run_catboost.py
```

4. (Optional) Tune LightGBM with Optuna (uses preprocessed artifacts):

```bash
python3 tune_lgbm.py
```

## 📁 Key files
- `run_catboost.py` — full preprocessing, TF‑IDF, optional SBERT embeddings, CatBoost + LightGBM training, stacking, and submission export.
- `tune_lgbm.py` — Optuna tuning script for LightGBM (saves `artifacts/lgb_best_params.json`).
- `Train.csv`, `Test.csv`, `SampleSubmission.csv` — dataset files.
- `artifacts/` — holds preprocessed pickles after running `run_catboost.py`: `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `lgb_best_params.json`.
- `*.csv` — generated submissions (e.g., `stacked_submission2.csv`, `ensemble_submission3.csv`).

## 🌐 Notes
- SBERT models are downloaded from Hugging Face; set `HF_TOKEN` in environment if you need authenticated downloads or higher rate limits.
- The pipeline fits TF‑IDF and group aggregations on TRAIN only to avoid leakage.
- Consider reducing SBERT dimensionality (PCA/SVD) before LightGBM to reduce model training time.

## ✅ Recommended next steps
- Run SBERT dimensionality reduction and re-evaluate LightGBM/CatBoost performance.
- Add a `requirements.txt` for reproducible environments.

---

If you want, I can commit & push this README (and other pending changes) to GitHub and create a short PR or push directly to `main`.