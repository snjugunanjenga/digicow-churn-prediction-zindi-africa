import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import optuna

# Load preprocessed data
X = pd.read_pickle('artifacts/X_train.pkl')
y = pd.read_pickle('artifacts/y_train.pkl')
# choose target to tune (07-day) -- tuning for all targets can be repeated
target = 'adopted_within_07_days'

y_target = y[target]

# Identify categorical cols (strings)
categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
for c in categorical:
    X[c] = X[c].astype('category')

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def objective(trial):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'n_jobs': -1
    }

    aucs = []
    for tr_idx, val_idx in skf.split(X, y_target):
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y_target.iloc[tr_idx], y_target.iloc[val_idx]
        # ensure categorical dtype
        for c in categorical:
            X_tr[c] = X_tr[c].astype('category')
            X_val[c] = X_val[c].astype('category')
        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                # early stopping
                __import__('lightgbm').early_stopping(50)
            ]
        )
        pred = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, pred))

    return float(np.mean(aucs))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print('Best trial:')
print(study.best_trial.params)

# Save best params
best_params = study.best_trial.params
with open('artifacts/lgb_best_params.json', 'w') as f:
    json.dump(best_params, f)
print('Saved best LightGBM params to artifacts/lgb_best_params.json')
