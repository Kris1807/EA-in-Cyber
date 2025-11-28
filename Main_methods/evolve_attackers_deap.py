import warnings

# Hide the LightGBM/sklearn feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import numpy as np
import random
from deap import base, creator, tools, algorithms
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import joblib


# ----------------------------------------------------------------------
#  PATHS + DATA LOADING
# ----------------------------------------------------------------------

# Make sure we can import thrember from General/Example
ROOT_DIR = Path(__file__).resolve().parent.parent  # EA-in-Cyber
sys.path.insert(0, str(ROOT_DIR / "General" / "Example"))
import thrember  # noqa: E402


def load_ember_malware(subset_size=5000):
    """
    Load only malware samples (y == 1) from EMBER2024 using thrember.
    Reads from EA-in-Cyber/General/Example/data where run.py wrote the .dat files.
    """
    data_dir = ROOT_DIR / "General" / "Example" / "data"
    print(f"Loading EMBER2024 malware from: {data_dir}")

    try:
        X_train, y_train = thrember.read_vectorized_features(str(data_dir), subset="train")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # Filter malware only
    malware_mask = (y_train == 1)
    X_malware = X_train[malware_mask]

    print(f"  Total malware samples: {len(X_malware)}")

    # Optionally subsample for speed
    if len(X_malware) > subset_size:
        idx = np.random.choice(len(X_malware), subset_size, replace=False)
        X_malware = X_malware[idx]

    print(f"  Using {len(X_malware)} malware samples")
    return X_malware, np.ones(len(X_malware), dtype=int)


def load_defender_model():
    """
    Load a fixed defender model trained earlier, e.g. baseline_lgbm_model.pkl.
    """
    model_path = ROOT_DIR / "models" / "baseline_lgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Defender model not found at {model_path}. "
            "Train it first with train_and_test.py."
        )
    print(f"Loading defender model from: {model_path}")
    clf = joblib.load(model_path)
    if not isinstance(clf, LGBMClassifier):
        print("Warning: loaded model is not an LGBMClassifier (but proceeding anyway).")
    return clf


# ----------------------------------------------------------------------
#  ATTACKER REPRESENTATION + FITNESS
# ----------------------------------------------------------------------

# Global variables that evaluation will use
FEATURE_INDICES = None   # np.array of indices we can perturb
EPS = 0.5                # magnitude of per-feature perturbation
BATCH_SIZE = 256         # how many malware samples per fitness evaluation


def evaluate_attacker(individual, X_malware, defender):
    """
    individual: list of length K with values in [-1, 1].
    We scale by EPS and add to selected feature indices.
    Fitness: negative evasion rate (we minimize), i.e. - (fraction of malware
    classified as benign by the defender).
    """
    try:
        K = len(individual)
        assert FEATURE_INDICES is not None, "FEATURE_INDICES not initialized"

        # Sample a batch of malware examples for this evaluation
        if len(X_malware) <= BATCH_SIZE:
            X_batch = X_malware
        else:
            idx = np.random.choice(len(X_malware), BATCH_SIZE, replace=False)
            X_batch = X_malware[idx]

        X_adv = X_batch.copy()

        # Build perturbation vector for those K features
        deltas = np.array(individual, dtype=float) * EPS  # scale [-1,1] -> [-EPS, EPS]

        # Apply same perturbation to all samples on those feature indices
        X_adv[:, FEATURE_INDICES] += deltas

        # Clip to non-negative (EMBER features shouldn't go negative)
        X_adv = np.clip(X_adv, 0, None)

        # Defender predictions: probability of malware (class 1)
        proba_mal = defender.predict_proba(X_adv)[:, 1]

        # Evasion = classified as benign (proba < 0.5)
        benign_pred = (proba_mal < 0.5).astype(int)
        evasion_rate = benign_pred.mean()

        # We want to MAXIMIZE evasion_rate -> MINIMIZE negative evasion_rate
        return (-evasion_rate,)

    except Exception as e:
        print(f"Error in attacker eval: {e}")
        # Very bad fitness if something breaks
        return (1.0,)


# ----------------------------------------------------------------------
#  MAIN: GA FOR ATTACKERS
# ----------------------------------------------------------------------

def main():
    print("Loading malware data...")
    X_malware, _ = load_ember_malware(subset_size=5000)
    if X_malware is None:
        print("Failed to load malware data. Exiting.")
        return

    print("Loading defender model...")
    defender = load_defender_model()

    n_features = X_malware.shape[1]
    print(f"Feature dimension: {n_features}")

    global FEATURE_INDICES
    K = 50  # number of features the attacker can perturb
    rng = np.random.default_rng(42)
    FEATURE_INDICES = rng.choice(n_features, size=K, replace=False)
    print(f"Attacker can perturb {K} features (indices saved in FEATURE_INDICES).")

    # ---------------- DEAP setup ----------------
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Each gene in [-1, 1]; later scaled by EPS in evaluation
    toolbox.register("gene", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, K)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness: attacker tries to maximize evasion (we minimize its negative)
    toolbox.register("evaluate",
                     lambda ind: evaluate_attacker(ind, X_malware, defender))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.3, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Keep genes in [-1, 1] after crossover/mutation
    def clip_bounds(min_val, max_val):
        def decorator(func):
            def wrapper(*args, **kwargs):
                offspring = func(*args, **kwargs)
                for child in offspring:
                    for i in range(len(child)):
                        child[i] = max(min(child[i], max_val), min_val)
                return offspring
            return wrapper
        return decorator

    toolbox.decorate("mate", clip_bounds(-1.0, 1.0))
    toolbox.decorate("mutate", clip_bounds(-1.0, 1.0))

    # ---------------- Run GA ----------------
    pop_size = 400
    n_gen = 100

    print("\nStarting attacker genetic algorithm...")
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.8,
        mutpb=0.5,
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    # ---------------- Results ----------------
    print("\n" + "=" * 60)
    print("ATTACKER EVOLUTION COMPLETE")
    print("=" * 60)

    best = hof[0]
    best_evasion = -best.fitness.values[0]   # we stored negative evasion

    print(f"\nBest attacker found:")
    print(f"  Evasion rate against defender: {best_evasion:.4f}")
    print(f"  First 10 perturbation genes: {best[:10]}")

    # You can later save best + FEATURE_INDICES if you want to reuse the attack
    # e.g. np.save(ROOT_DIR/'results'/'best_attacker.npy', np.array(best))


if __name__ == "__main__":
    main()
