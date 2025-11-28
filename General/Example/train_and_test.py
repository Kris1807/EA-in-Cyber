import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import thrember
from lightgbm import LGBMClassifier

# NEW: imports for saving the model
from pathlib import Path
import joblib

# Load data
X_train, y_train = thrember.read_vectorized_features("data", subset="train")
X_test,  y_test  = thrember.read_vectorized_features("data", subset="test")

# Small validation split from train
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    stratify=y_train,
    random_state=42
)

# Train a baseline model
clf = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.9,
    colsample_bytree=0.9,
    n_jobs=-1
)
clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc")

# Evaluate
proba_val = clf.predict_proba(X_val)[:, 1]
proba_test = clf.predict_proba(X_test)[:, 1]
pred_test  = (proba_test >= 0.5).astype(int)

print("VAL AUC:", roc_auc_score(y_val, proba_val))
print("TEST AUC:", roc_auc_score(y_test, proba_test))
print("TEST ACC:", accuracy_score(y_test, pred_test))

tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
print(f"Confusion Matrix (test) -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

# ---------- SAVE DEFENDER MODEL FOR ATTACKER SCRIPT ----------

# This file lives in: EA-in-Cyber/General/Example/train_and_test.py
BASE = Path(__file__).resolve().parent           # .../EA-in-Cyber/General/Example
ROOT_DIR = BASE.parent.parent                    # .../EA-in-Cyber
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "baseline_lgbm_model.pkl"
joblib.dump(clf, model_path)
print(f"Defender model saved to: {model_path}")
