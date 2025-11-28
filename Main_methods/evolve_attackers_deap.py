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


def load_defender_pool():
    """
    Load all defender models from models/random_defenders/.
    """
    pool_dir = ROOT_DIR / "models" / "random_defenders"
    if not pool_dir.exists():
        raise FileNotFoundError(f"Defender pool dir not found: {pool_dir}")

    pkls = sorted(pool_dir.glob("defender_*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No defender_*.pkl found in {pool_dir}")

    print(f"Loading {len(pkls)} defenders from: {pool_dir}")
    defenders = []
    for p in pkls:
        clf = joblib.load(p)
        defenders.append(clf)
        print(f"  loaded {p.name}")
    return defenders


# ----------------------------------------------------------------------
#  ATTACKER REPRESENTATION + FITNESS
# ----------------------------------------------------------------------

# Global variables that evaluation will use
FEATURE_INDICES = None   # np.array of indices we can perturb
EPS = 0.5                # magnitude of per-feature perturbation
BATCH_SIZE = 256         # how many malware samples per fitness evaluation


def evaluate_attacker(individual, X_malware, defenders):
    """
    individual: list of length K with values in [-1, 1].
    Fitness: negative average evasion rate across all defenders.
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
        deltas = np.array(individual, dtype=float) * EPS  # [-EPS, EPS]
        X_adv[:, FEATURE_INDICES] += deltas
        X_adv = np.clip(X_adv, 0, None)

        evasion_rates = []
        for clf in defenders:
            proba_mal = clf.predict_proba(X_adv)[:, 1]
            benign_pred = (proba_mal < 0.5).astype(int)
            evasion_rates.append(benign_pred.mean())

        # average evasion across defenders
        avg_evasion = float(np.mean(evasion_rates))

        # maximize evasion -> minimize negative
        return (-avg_evasion,)

    except Exception as e:
        print(f"Error in attacker eval: {e}")
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

    print("Loading defender pool...")
    defenders = load_defender_pool()

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
                     lambda ind: evaluate_attacker(ind, X_malware, defenders))
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
    pop_size = 100
    n_gen = 20

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
