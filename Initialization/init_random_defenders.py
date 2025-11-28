"""
Creates N randomly initialized defender models (LightGBM classifiers)
with random hyperparameters, and stores them in a list.

Optionally saves them to disk as defender_00.pkl, defender_01.pkl, ...
"""

import random
from pathlib import Path
import joblib
from lightgbm import LGBMClassifier


# ---------------------------------------
# CONFIG
# ---------------------------------------

N_DEFENDERS = 30   # number of defender models to generate

# Where to save (inside EA-in-Cyber/models/random_defenders/)
BASE = Path(__file__).resolve().parent.parent  # EA-in-Cyber/
SAVE_DIR = BASE / "models" / "random_defenders"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------
# RANDOM HYPERPARAMETERS
# ---------------------------------------

def random_defender():
    """Create one defender model with random hyperparameters."""
    return LGBMClassifier(
        n_estimators=random.randint(50, 800),
        learning_rate=random.uniform(0.001, 0.2),
        num_leaves=random.randint(8, 128),
        subsample=random.uniform(0.5, 1.0),
        colsample_bytree=random.uniform(0.5, 1.0),
        n_jobs=-1
    )


# ---------------------------------------
# MAIN
# ---------------------------------------

def main():
    defenders = []

    print(f"Creating {N_DEFENDERS} random defender models...")

    for i in range(N_DEFENDERS):
        model = random_defender()
        defenders.append(model)

        # Optional: save each model (untrained)
        model_path = SAVE_DIR / f"defender_{i:02d}.pkl"
        joblib.dump(model, model_path)

        print(f"  → defender_{i:02d} created and saved.")

    print("\nDone!")
    print(f"Saved all defenders to: {SAVE_DIR}")

    return defenders


if __name__ == "__main__":
    main()