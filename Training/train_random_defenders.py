import joblib
from pathlib import Path
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import sys

# Make thrember importable from General/Example
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "General" / "Example"))
import thrember  # noqa: E402

MODELS_DIR = ROOT_DIR / "models" / "random_defenders"


def load_train_data():
    data_dir = ROOT_DIR / "General" / "Example" / "data"
    print(f"Loading EMBER train data from: {data_dir}")
    X_train, y_train = thrember.read_vectorized_features(str(data_dir), subset="train")
    # small split just for sanity (we mostly care about training here)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    return X_tr, y_tr, X_val, y_val


def main():
    X_tr, y_tr, X_val, y_val = load_train_data()

    if not MODELS_DIR.exists():
        print(f"No random_defenders directory at {MODELS_DIR}")
        return

    pkls = sorted(MODELS_DIR.glob("defender_*.pkl"))
    if not pkls:
        print(f"No defender_*.pkl found in {MODELS_DIR}")
        return

    print(f"Training {len(pkls)} random defenders...")

    for path in pkls:
        print(f"\nTraining {path.name} ...")
        clf: LGBMClassifier = joblib.load(path)
        clf.fit(X_tr, y_tr)
        # overwrite with trained version
        joblib.dump(clf, path)
        print(f"  Saved trained defender to: {path}")

    print("\nAll defenders trained.")


if __name__ == "__main__":
    main()