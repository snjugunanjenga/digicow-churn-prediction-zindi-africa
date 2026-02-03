import pandas as pd
import numpy as np
import ast
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Utility / preprocessing
# -----------------------------

def safe_parse_topics(x):
    """Parse topics which may be stored as list-string or malformed text.
    Returns a list of cleaned topic tokens (lowercased, punctuation removed).
    """
    if pd.isna(x):
        return []
    try:
        lst = ast.literal_eval(x)
        if not isinstance(lst, (list, tuple)):
            lst = [str(lst)]
    except Exception:
        # fallback: treat as raw string
        lst = [x]

    tokens = []
    for item in lst:
        if not isinstance(item, str):
            item = str(item)
        # split on commas because some items are comma-separated inside list entries
        for part in item.split(','):
            t = re.sub(r"[^0-9a-zA-Z ]+", " ", part).strip().lower()
            if t:
                tokens.append(t)
    # dedupe while preserving order
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')
sample_submission_df = pd.read_csv('SampleSubmission.csv')

train_df.set_index('ID', inplace=True)
test_df.set_index('ID', inplace=True)

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Keep a copy of targets
targets = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
y = train_df[targets]

# Concatenate for joint preprocessing
combined = pd.concat([train_df.drop(columns=targets), test_df], axis=0)

# -----------------------------
# Feature engineering
# -----------------------------

# Parse topics into clean token lists and build a text field
combined['topics_list_parsed'] = combined['topics_list'].apply(safe_parse_topics)
combined['topics_text'] = combined['topics_list_parsed'].apply(lambda x: ' '.join([t.replace(' ', '_') for t in x]))
combined['topic_count'] = combined['topics_list_parsed'].apply(len)

# Date features
combined['first_training_date'] = pd.to_datetime(combined['first_training_date'], errors='coerce')
combined['month'] = combined['first_training_date'].dt.month.fillna(0).astype(int)
combined['dayofweek'] = combined['first_training_date'].dt.dayofweek.fillna(0).astype(int)
combined['year'] = combined['first_training_date'].dt.year.fillna(0).astype(int)
combined.drop(columns=['first_training_date', 'topics_list'], inplace=True)

# Numeric derived features
combined['repeat_ratio'] = combined['num_repeat_trainings'] / (combined['num_total_trainings'] + 1)
combined['trainer_diversity'] = combined['num_unique_trainers'] / (combined['num_total_trainings'] + 1)
combined['has_second_training'] = combined['has_second_training'].astype(int)
combined['days_to_second_training_missing'] = combined['days_to_second_training'].isnull().astype(int)
combined['days_to_second_training'] = combined['days_to_second_training'].fillna(-1)

# Map age to numeric
combined['age'] = combined['age'].map({'Below 35': 0, 'Above 35': 1})

# Aggregations computed on TRAIN only to avoid leakage into training features
train_index = train_df.index
for g in ['county', 'subcounty', 'ward']:
    agg = combined.loc[train_index].groupby(g)['age'].mean()
    combined[f'{g}_age_mean'] = combined[g].map(agg)
    # If a group in test wasn't seen in train, fall back to overall train mean
    combined[f'{g}_age_mean'] = combined[f'{g}_age_mean'].fillna(train_df['age'].map({'Below 35': 0, 'Above 35': 1}).mean())

# -----------------------------
# Topic TF-IDF (fit on TRAIN data only to avoid leakage)
# -----------------------------
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

# Word-level TF-IDF (unigrams..trigrams) + char-level TF-IDF (char_wb)
word_vectorizer = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 3),
    min_df=2,
    sublinear_tf=True,
    token_pattern=r'(?u)\b\w+\b'
)
char_vectorizer = TfidfVectorizer(
    max_features=200,
    analyzer='char_wb',
    ngram_range=(3, 5),
    sublinear_tf=True
)

train_topics = combined.loc[train_df.index, 'topics_text'].fillna('')
# fit only on train to avoid leakage
word_vectorizer.fit(train_topics)
char_vectorizer.fit(train_topics)

# transform all rows
w_tfidf = word_vectorizer.transform(combined['topics_text'].fillna(''))
c_tfidf = char_vectorizer.transform(combined['topics_text'].fillna(''))

tfidf_all = hstack([w_tfidf, c_tfidf])

# Reduce dimensionality with SVD (fit on train subset)
n_svd = 100
train_mask = combined.index.isin(train_df.index)
# ensure n_components < n_features
n_svd_eff = min(n_svd, tfidf_all.shape[1]-1)
if n_svd_eff <= 0:
    # fallback: use raw TF-IDF if too few features
    tfidf_dense = tfidf_all.toarray()
    cols = [f'topic_tfidf_{i}' for i in range(tfidf_dense.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_dense, index=combined.index, columns=cols)
else:
    svd = TruncatedSVD(n_components=n_svd_eff, random_state=42)
    svd.fit(tfidf_all[train_mask])
    tfidf_svd = svd.transform(tfidf_all)
    cols = [f'topic_tfidf_svd_{i}' for i in range(tfidf_svd.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_svd, index=combined.index, columns=cols)

combined = pd.concat([combined, tfidf_df], axis=1)
print('TF-IDF (word+char) fit on train and SVD applied. Result dims:', tfidf_df.shape)

# -----------------------------
# Optional: Sentence-BERT embeddings for topics_text
# -----------------------------
try:
    from sentence_transformers import SentenceTransformer
    print('Generating SBERT embeddings (all-MiniLM-L6-v2)...')
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    # Use the topics_text column before dropping
    sbert_emb = sbert.encode(combined['topics_text'].fillna(' ').tolist(), show_progress_bar=True)
    sbert_df = pd.DataFrame(sbert_emb, index=combined.index, columns=[f'topic_sbert_{i}' for i in range(sbert_emb.shape[1])])
    combined = pd.concat([combined, sbert_df], axis=1)
    print('SBERT embeddings added, shape:', sbert_df.shape)
except Exception as e:
    print('SBERT embeddings skipped due to error:', e)

# Drop intermediate topic columns now that embeddings & TF-IDF are created
combined.drop(columns=['topics_text', 'topics_list_parsed'], inplace=True)
print('Feature engineering complete. Shape:', combined.shape)

# -----------------------------
# Split back to train / test
# -----------------------------
X = combined.loc[train_df.index].copy()
X_test = combined.loc[test_df.index].copy()

# Save preprocessed datasets for tuning or reuse
import os
os.makedirs('artifacts', exist_ok=True)
X.to_pickle('artifacts/X_train.pkl')
X_test.to_pickle('artifacts/X_test.pkl')
train_df[targets].to_pickle('artifacts/y_train.pkl')
# Also save LightGBM-ready copies (categoricals as categories will be created later)
print('Saved preprocessed datasets to artifacts/')

# Identify categorical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
# convert to string for CatBoost
X[categorical_features] = X[categorical_features].astype(str)
X_test[categorical_features] = X_test[categorical_features].astype(str)

# Align columns
X_test = X_test[X.columns]

# Prepare LightGBM-compatible datasets (pandas categorical dtypes)
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in categorical_features:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# -----------------------------
# Training with Stratified K-Fold and early stopping
# -----------------------------

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

final_test_preds = {t: np.zeros(X_test.shape[0]) for t in targets}
oof_preds = {t: np.zeros(X.shape[0]) for t in targets}
# For LightGBM baseline and simple ensemble
final_test_preds_lgb = {t: np.zeros(X_test.shape[0]) for t in targets}
oof_preds_lgb = {t: np.zeros(X.shape[0]) for t in targets}

for target in targets:
    print(f'\nTraining for target: {target}')
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y[target])):
        print(f' Fold {fold + 1}/{n_splits}')
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[target].iloc[tr_idx], y[target].iloc[val_idx]

        # compute class weights to help imbalance
        unique_classes = np.unique(y_tr)
        if len(unique_classes) < 2:
            # all samples are of a single class in this fold -> fallback to uniform weights
            cw_list = [1.0, 1.0]
        else:
            cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
            cw_list = [float(cw[0]), float(cw[1])]

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            verbose=100,
            class_weights=cw_list
        )

        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            cat_features=categorical_features,
            use_best_model=True
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[target][val_idx] = val_pred
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f'  CatBoost Fold AUC: {fold_auc:.4f}')

        # test predictions (averaged)
        final_test_preds[target] += model.predict_proba(X_test)[:, 1] / n_splits

        # ------------------ LightGBM baseline (same fold) ------------------
        # Try to load tuned LightGBM params if available
        import json, os
        tuned_params = {}
        try:
            if os.path.exists('artifacts/lgb_best_params.json'):
                with open('artifacts/lgb_best_params.json', 'r') as f:
                    tuned_params = json.load(f)
                print('Loaded tuned LightGBM params from artifacts/lgb_best_params.json')
        except Exception:
            tuned_params = {}

        lgb_params = {
            'n_estimators': 1000,
            'learning_rate': tuned_params.get('learning_rate', 0.05),
            'num_leaves': int(tuned_params.get('num_leaves', 31)),
            'max_depth': int(tuned_params.get('max_depth', -1)),
            'min_child_samples': int(tuned_params.get('min_child_samples', 20)),
            'subsample': tuned_params.get('subsample', 1.0),
            'colsample_bytree': tuned_params.get('colsample_bytree', 1.0),
            'reg_alpha': tuned_params.get('reg_alpha', 0.0),
            'reg_lambda': tuned_params.get('reg_lambda', 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        lgb_model = LGBMClassifier(**lgb_params)
        # LightGBM accepts list of categorical feature names
        try:
            # Use versions with proper categorical dtypes for LightGBM
            X_tr_l = X_lgb.iloc[tr_idx]
            X_val_l = X_lgb.iloc[val_idx]
            try:
                lgb_model.fit(
                    X_tr_l, y_tr,
                    eval_set=[(X_val_l, y_val)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                    categorical_feature=categorical_features
                )
            except Exception:
                # Fall back to training without explicit categorical feature list
                lgb_model.fit(
                    X_tr_l, y_tr,
                    eval_set=[(X_val_l, y_val)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(50)],
                )
        except Exception:
            # Fall back to fit without explicit cat features
            lgb_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(50)],
            )

        lgb_val_pred = lgb_model.predict_proba(X_val_l)[:, 1]
        oof_preds_lgb[target][val_idx] = lgb_val_pred
        fold_auc_lgb = roc_auc_score(y_val, lgb_val_pred)
        print(f'  LightGBM Fold AUC: {fold_auc_lgb:.4f}')

        final_test_preds_lgb[target] += lgb_model.predict_proba(X_test_lgb)[:, 1] / n_splits

    print(f' >> CatBoost CV AUC for {target}: {np.mean(fold_scores):.4f} +- {np.std(fold_scores):.4f}')
    # compute LGBM fold-wise AUCs
    lgb_fold_mean = roc_auc_score(y[target], oof_preds_lgb[target]) if np.any(oof_preds_lgb[target]) else np.nan
    print(f' >> LightGBM OOF AUC for {target}: {lgb_fold_mean:.4f}')

# Combined validation metric (mean of targets) for CatBoost, LightGBM, and ensemble
mean_aucs_cb = [roc_auc_score(y[t], oof_preds[t]) for t in targets]
mean_aucs_lgb = [roc_auc_score(y[t], oof_preds_lgb[t]) for t in targets]
print('\nOOF AUCs per target (CatBoost vs LightGBM vs Ensemble avg):')
for t, a_cb, a_lgb in zip(targets, mean_aucs_cb, mean_aucs_lgb):
    ensemble_oof = roc_auc_score(y[t], (oof_preds[t] + oof_preds_lgb[t]) / 2)
    print(f' - {t}: CatBoost={a_cb:.4f}, LightGBM={a_lgb:.4f}, Ensemble={ensemble_oof:.4f}')
print(f'Overall mean OOF AUC (CatBoost): {np.mean(mean_aucs_cb):.4f}')
print(f'Overall mean OOF AUC (LightGBM): {np.mean(mean_aucs_lgb):.4f}')
print(f'Overall mean OOF AUC (Ensemble avg): {np.mean([roc_auc_score(y[t], (oof_preds[t] + oof_preds_lgb[t])/2) for t in targets]):.4f}')

# -----------------------------
# Build submissions (CatBoost-only and simple mean ensemble)
# -----------------------------
# CatBoost-only submission (kept for reference)
submission_cb = pd.DataFrame(index=X_test.index)
submission_cb['Target_07_AUC'] = final_test_preds['adopted_within_07_days']
submission_cb['Target_90_AUC'] = final_test_preds['adopted_within_90_days']
submission_cb['Target_120_AUC'] = final_test_preds['adopted_within_120_days']
submission_cb['Target_07_LogLoss'] = final_test_preds['adopted_within_07_days']
submission_cb['Target_90_LogLoss'] = final_test_preds['adopted_within_90_days']
submission_cb['Target_120_LogLoss'] = final_test_preds['adopted_within_120_days']
submission_cb = submission_cb[sample_submission_df.columns[1:]]
submission_cb = submission_cb.reindex(sample_submission_df['ID']).fillna(0)
submission_cb.to_csv('catboost_submission_V8.csv')

# Ensemble (mean of CatBoost + LightGBM)
submission_ens = pd.DataFrame(index=X_test.index)
submission_ens['Target_07_AUC'] = (final_test_preds['adopted_within_07_days'] + final_test_preds_lgb['adopted_within_07_days']) / 2
submission_ens['Target_90_AUC'] = (final_test_preds['adopted_within_90_days'] + final_test_preds_lgb['adopted_within_90_days']) / 2
submission_ens['Target_120_AUC'] = (final_test_preds['adopted_within_120_days'] + final_test_preds_lgb['adopted_within_120_days']) / 2
submission_ens['Target_07_LogLoss'] = submission_ens['Target_07_AUC']
submission_ens['Target_90_LogLoss'] = submission_ens['Target_90_AUC']
submission_ens['Target_120_LogLoss'] = submission_ens['Target_120_AUC']
submission_ens = submission_ens[sample_submission_df.columns[1:]]
submission_ens = submission_ens.reindex(sample_submission_df['ID']).fillna(0)
submission_ens.to_csv('ensemble_submission3.csv')

# -----------------------------
# Stacking meta-learner (Logistic Regression)
# -----------------------------
print('\nTraining stacking meta-learner (LogisticRegression) on OOF predictions...')
stacked_test_preds = {}
stacked_oof_preds = {}
for target in targets:
    # build meta features (columns: catboost_oof, lgb_oof)
    meta_X = np.vstack([oof_preds[target], oof_preds_lgb[target]]).T
    meta_test = np.vstack([final_test_preds[target], final_test_preds_lgb[target]]).T

    # train logistic regression on OOF preds
    lr = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    lr.fit(meta_X, y[target])

    meta_oof_pred = lr.predict_proba(meta_X)[:, 1]
    meta_test_pred = lr.predict_proba(meta_test)[:, 1]

    stacked_oof_preds[target] = meta_oof_pred
    stacked_test_preds[target] = meta_test_pred

    print(f' Stack OOF AUC for {target}: {roc_auc_score(y[target], meta_oof_pred):.4f}')

# overall metrics
stack_mean = np.mean([roc_auc_score(y[t], stacked_oof_preds[t]) for t in targets])
ensemble_mean = np.mean([roc_auc_score(y[t], (oof_preds[t] + oof_preds_lgb[t]) / 2) for t in targets])
print(f'\nOverall mean OOF AUC (Ensemble avg): {ensemble_mean:.4f}')
print(f'Overall mean OOF AUC (Stack): {stack_mean:.4f}')

# Save stacked submission
submission_stack = pd.DataFrame(index=X_test.index)
submission_stack['Target_07_AUC'] = stacked_test_preds['adopted_within_07_days']
submission_stack['Target_90_AUC'] = stacked_test_preds['adopted_within_90_days']
submission_stack['Target_120_AUC'] = stacked_test_preds['adopted_within_120_days']
submission_stack['Target_07_LogLoss'] = submission_stack['Target_07_AUC']
submission_stack['Target_90_LogLoss'] = submission_stack['Target_90_AUC']
submission_stack['Target_120_LogLoss'] = submission_stack['Target_120_AUC']
submission_stack = submission_stack[sample_submission_df.columns[1:]]
submission_stack = submission_stack.reindex(sample_submission_df['ID']).fillna(0)
submission_stack.to_csv('stacked_submission2.csv')

# Choose the best submission (stack vs ensemble)
if stack_mean > ensemble_mean:
    chosen = 'stacked_submission.csv'
    print('\nStacked submission outperformed simple ensemble; keeping `stacked_submission.csv`.')
else:
    chosen = 'ensemble_submission.csv'
    print('\nSimple ensemble is better or equal; keeping `ensemble_submission.csv`.')

print(f'Chosen submission file: {chosen}')
print('\nSaved `stacked_submission.csv` and `ensemble_submission2.csv`')
print(submission_stack.head())