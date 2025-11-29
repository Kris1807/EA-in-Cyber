import warnings

# Hide the LightGBM/sklearn feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import numpy as np
import random
from deap import base, creator, tools, algorithms
from pathlib import Path
import sys
import joblib

# ----------------------------------------------------------------------
#  PATHS + DATA LOADING
# ----------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent  # EA-in-Cyber
RESULTS_DIR = ROOT_DIR / "results"
EVOLVED_ATTACKERS_FILE = RESULTS_DIR / "evolved_attackers.pkl"

# Make sure we can import thrember from General/Example
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

    malware_mask = (y_train == 1)
    X_malware = X_train[malware_mask]

    print(f"  Total malware samples: {len(X_malware)}")

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


def load_evolved_attackers():
    """Load the persistent list of best attackers (if any)."""
    if EVOLVED_ATTACKERS_FILE.exists():
        return joblib.load(EVOLVED_ATTACKERS_FILE)
    return []


def save_evolved_attackers(evolved_attackers):
    """Save the persistent list of best attackers."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(evolved_attackers, EVOLVED_ATTACKERS_FILE)


# ----------------------------------------------------------------------
#  ATTACKER REPRESENTATION + EVASION COMPUTATION
# ----------------------------------------------------------------------

FEATURE_INDICES = None   # np.array of indices we can perturb
EPS = 0.5                # magnitude of per-feature perturbation
BATCH_SIZE = 256         # how many malware samples per fitness evaluation


def compute_evasion(genes, feat_idx, X_malware, defenders):
    """
    Core routine: given genes and feature indices, compute average evasion
    across defenders on a sampled batch of malware.
    """
    if len(X_malware) <= BATCH_SIZE:
        X_batch = X_malware
    else:
        idx = np.random.choice(len(X_malware), BATCH_SIZE, replace=False)
        X_batch = X_malware[idx]

    X_adv = X_batch.copy()
    deltas = np.array(genes, dtype=float) * EPS  # [-EPS, EPS]
    X_adv[:, feat_idx] += deltas
    X_adv = np.clip(X_adv, 0, None)

    evasion_rates = []
    for clf in defenders:
        proba_mal = clf.predict_proba(X_adv)[:, 1]
        benign_pred = (proba_mal < 0.5).astype(int)
        evasion_rates.append(benign_pred.mean())

    return float(np.mean(evasion_rates))


def recompute_evasion_for_attacker(attacker, X_malware, defenders):
    """
    Recompute evasion of a stored attacker dict against the current
    defender pool, using its own feature_indices and genes.
    attacker: {"evasion": float, "genes": [...], "feature_indices": [...]}
    """
    genes = attacker["genes"]
    feat_idx = np.array(attacker["feature_indices"], dtype=int)
    return compute_evasion(genes, feat_idx, X_malware, defenders)


def evaluate_attacker(individual, X_malware, defenders):
    """
    individual: list of length K with values in [-1, 1].
    Fitness: negative average evasion rate across all defenders.
    """
    try:
        assert FEATURE_INDICES is not None, "FEATURE_INDICES not initialized"
        avg_evasion = compute_evasion(individual, FEATURE_INDICES, X_malware, defenders)
        return (-avg_evasion,)  # DEAP minimizes
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

    # STEP 1: load old best attackers and re-evaluate them
    old_attackers = load_evolved_attackers()
    if old_attackers:
        print(f"\nRe-evaluating {len(old_attackers)} stored attackers "
              f"against current defender pool...")
        for atk in old_attackers:
            new_evasion = recompute_evasion_for_attacker(atk, X_malware, defenders)
            atk["evasion"] = new_evasion
    else:
        print("\nNo stored attackers found (first run).")

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
    toolbox.register("gene", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, K)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate",
                     lambda ind: evaluate_attacker(ind, X_malware, defenders))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.3, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

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
    best_evasion = -best.fitness.values[0]

    print(f"\nBest attacker found:")
    print(f"  Evasion rate against defender: {best_evasion:.4f}")
    print(f"  First 10 perturbation genes: {best[:10]}")

    # STEP 2: merge old attackers + new population, keep top 1000
    evolved_attackers = list(old_attackers)  # copy

    for ind in pop:
        evasion = -ind.fitness.values[0]
        evolved_attackers.append({
            "evasion": evasion,
            "genes": list(ind),
            "feature_indices": FEATURE_INDICES.tolist(),
        })

    evolved_attackers.sort(key=lambda x: x["evasion"], reverse=True)
    evolved_attackers = evolved_attackers[:1000]

    save_evolved_attackers(evolved_attackers)
    print(f"\nUpdated evolved_attackers (top {len(evolved_attackers)}) "
          f"and saved to {EVOLVED_ATTACKERS_FILE}")


if __name__ == "__main__":
    main()
