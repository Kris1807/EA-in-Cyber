import warnings
import joblib # type: ignore

# Only way to get rid of the warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import numpy as np
import random
from deap import base, creator, tools, algorithms
from lightgbm import LGBMClassifier # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from pathlib import Path
import sys

# ----------------------------------------------------------------------
# PATHS / GLOBALS
# ----------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent   # .../EA-in-Cyber
MODELS_DIR = ROOT_DIR / "models" / "random_defenders"

RESULTS_DIR = ROOT_DIR / "results"
EVOLVED_ATTACKERS_FILE = RESULTS_DIR / "evolved_attackers.pkl"
EVOLVED_DEFENDERS_FILE = RESULTS_DIR / "evolved_defenders.pkl"

# Attack-side constants (must match attacker script)
EPS = 0.5
ADV_BATCH_SIZE = 256  # batch size for adversarial evaluation

# Make thrember importable
sys.path.insert(0, str(ROOT_DIR / "General" / "Example"))
import thrember  # noqa: E402 # type: ignore


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------

def load_ember_data(subset="train", subset_size=5000):
    """
    Load EMBER2024 features using thrember from the .dat files
    in EA-in-Cyber/General/Example/data.
    """
    data_dir = ROOT_DIR / "General" / "Example" / "data"
    print(f"Loading EMBER2024 data from: {data_dir} (subset='{subset}')")

    try:
        X, y = thrember.read_vectorized_features(str(data_dir), subset=subset)

        if len(X) > subset_size:
            indices = np.random.choice(len(X), subset_size, replace=False)
            X = X[indices]
            y = y[indices]

        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


# ----------------------------------------------------------------------
# ATTACKERS / DEFENDERS PERSISTENT LISTS
# ----------------------------------------------------------------------

def load_top_attackers(max_attackers=20):
    """
    Load best attackers from results/evolved_attackers.pkl (if any),
    sorted by evasion descending, and return the top max_attackers.
    """
    if not EVOLVED_ATTACKERS_FILE.exists():
        print("No evolved attackers file found. Defenders will be evaluated on clean AUC only.")
        return []

    attackers = joblib.load(EVOLVED_ATTACKERS_FILE)
    if not attackers:
        print("evolved_attackers.pkl is empty. Defenders will be evaluated on clean AUC only.")
        return []

    attackers.sort(key=lambda a: a.get("evasion", 0.0), reverse=True)
    attackers = attackers[:max_attackers]

    print(f"Loaded {len(attackers)} top attackers from {EVOLVED_ATTACKERS_FILE}")
    return attackers


def load_evolved_defenders():
    """Load persistent list of best defenders (if exists)."""
    if EVOLVED_DEFENDERS_FILE.exists():
        return joblib.load(EVOLVED_DEFENDERS_FILE)
    return []


def save_evolved_defenders(evolved_defenders):
    """Save persistent list of best defenders."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(evolved_defenders, EVOLVED_DEFENDERS_FILE)


# ----------------------------------------------------------------------
# INITIAL DEFENDERS (FROM random_defenders MODELS)
# ----------------------------------------------------------------------

def load_initial_defender_hparams():
    """
    Load the 30 saved defenders from models/random_defenders and
    extract [n_estimators, learning_rate, num_leaves] for each.
    """
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Defender pool dir not found: {MODELS_DIR}")

    pkls = sorted(MODELS_DIR.glob("defender_*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No defender_*.pkl found in {MODELS_DIR}")

    print(f"Loading initial defender hyperparameters from {MODELS_DIR}")
    individuals = []
    for p in pkls:
        clf = joblib.load(p)
        params = clf.get_params()
        n_estimators = float(params.get("n_estimators", 400))
        learning_rate = float(params.get("learning_rate", 0.05))
        num_leaves = float(params.get("num_leaves", 64))
        ind = [n_estimators, learning_rate, num_leaves]
        individuals.append(ind)
        print(f"  {p.name}: n_estimators={n_estimators}, "
              f"lr={learning_rate}, num_leaves={num_leaves}")
    return individuals


# ----------------------------------------------------------------------
# DEFENDER SCORING vs ATTACKERS
# ----------------------------------------------------------------------

def defender_score_and_metrics(hyperparams, X_train, y_train, X_test, y_test, attackers):
    """
    Train a defender with given hyperparams and compute:

      - auc_clean: AUC on clean test data
      - robust_det: mean detection rate against adversarial malware generated
                    by the given attackers
      - score: combined scalar for GA (higher is better)

    Returns (score, auc_clean, robust_det).
    """
    n_estimators = int(hyperparams[0])
    learning_rate = float(hyperparams[1])
    num_leaves = max(2, int(hyperparams[2]))

    # Train defender
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # Clean AUC
    proba_clean = model.predict_proba(X_test)[:, 1]
    auc_clean = roc_auc_score(y_test, proba_clean)

    # If no attackers yet, just use clean AUC as score
    if not attackers:
        return auc_clean, auc_clean, 0.0

    # Build malware subset from test set
    mal_mask = (y_test == 1)
    X_mal = X_test[mal_mask]
    if len(X_mal) == 0:
        robust_det = 0.0
    else:
        # sample a batch for speed
        if len(X_mal) > ADV_BATCH_SIZE:
            idx = np.random.choice(len(X_mal), ADV_BATCH_SIZE, replace=False)
            X_batch = X_mal[idx]
        else:
            X_batch = X_mal

        det_rates = []
        for atk in attackers:
            genes = np.array(atk["genes"], dtype=float)
            feat_idx = np.array(atk["feature_indices"], dtype=int)

            X_adv = X_batch.copy()
            X_adv[:, feat_idx] += genes * EPS
            X_adv = np.clip(X_adv, 0, None)

            proba_adv = model.predict_proba(X_adv)[:, 1]
            benign_pred = (proba_adv < 0.5).astype(int)
            detection_rate = 1.0 - benign_pred.mean()  # 1 - evasion
            det_rates.append(detection_rate)

        robust_det = float(np.mean(det_rates))

    # Combine clean performance + robustness (tweak weights if you like)
    score = 0.5 * auc_clean + 0.5 * robust_det
    return score, auc_clean, robust_det


def evaluate_defender(hyperparams, X_train, y_train, X_test, y_test, attackers):
    """
    Wrapper for DEAP: returns a 1-tuple with the *negative* score,
    since DEAP minimizes by default.
    """
    try:
        score, _, _ = defender_score_and_metrics(
            hyperparams, X_train, y_train, X_test, y_test, attackers
        )
        return (-score,)
    except Exception as e:
        print(f"Error in eval: {e}")
        return (1.0,)


# ----------------------------------------------------------------------
# MAIN: GA FOR DEFENDERS
# ----------------------------------------------------------------------

def main():
    print("Loading EMBER2024 data...")
    X_train, y_train = load_ember_data(subset_size=5000)
    if X_train is None:
        print("Failed to load EMBER data. Exiting.")
        return

    print(f"Loaded {len(X_train)} samples with {X_train.shape[1]} features")

    # Small validation split from train
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

    # Load best attackers (if any)
    attackers = load_top_attackers(max_attackers=20)

    # Load existing evolved defenders (if any) and re-evaluate them
    evolved_defenders = load_evolved_defenders()
    if evolved_defenders:
        print(f"\nRe-evaluating {len(evolved_defenders)} stored defenders "
              f"against current attackers...")
        for d in evolved_defenders:
            hyper = [d["n_estimators"], d["learning_rate"], d["num_leaves"]]
            score, auc_clean, robust_det = defender_score_and_metrics(
                hyper, X_tr, y_tr, X_te, y_te, attackers
            )
            d["score"] = score
            d["auc_clean"] = auc_clean
            d["robust_detection"] = robust_det
    else:
        print("\nNo stored defenders found; starting from random_defenders.")

    # Initialize GA population
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Hyperparameter bounds
    toolbox.register("n_estimators", random.uniform, 100, 1000)
    toolbox.register("learning_rate", random.uniform, 0.001, 0.1)
    toolbox.register("num_leaves", random.uniform, 10, 255)

    def init_individual_from_list(values):
        return creator.Individual(values)

    toolbox.register("individual_from_list", init_individual_from_list)

    # Choose seed population:
    #   if we have evolved defenders, seed from them;
    #   otherwise, from initial random_defenders hyperparams.
    if evolved_defenders:
        print("Seeding GA population from evolved_defenders list.")
        initial_hparams = [
            [float(d["n_estimators"]), float(d["learning_rate"]), float(d["num_leaves"])]
            for d in evolved_defenders
        ]
    else:
        initial_hparams = load_initial_defender_hparams()

    pop = [toolbox.individual_from_list(h) for h in initial_hparams]
    pop_size = len(pop)
    print(f"\nInitialized population with {pop_size} defenders.")

    # GA operators
    toolbox.register(
        "evaluate",
        lambda ind: evaluate_defender(ind, X_tr, y_tr, X_te, y_te, attackers)
    )
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Bounds for mutation
    def check_bounds(min_vals, max_vals):
        def decorator(func):
            def wrapper(*args, **kwargs):
                offspring = func(*args, **kwargs)
                for child in offspring:
                    child[0] = np.clip(child[0], min_vals[0], max_vals[0])
                    child[1] = np.clip(child[1], min_vals[1], max_vals[1])
                    child[2] = np.clip(child[2], min_vals[2], max_vals[2])
                return offspring
            return wrapper
        return decorator

    toolbox.decorate("mate", check_bounds([100, 0.001, 10], [1000, 0.1, 255]))
    toolbox.decorate("mutate", check_bounds([100, 0.001, 10], [1000, 0.1, 255]))

    # Run GA
    print("\nStarting defender genetic algorithm...")
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.8,
        mutpb=0.5,
        ngen=100,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    # Results
    print("\n" + "=" * 60)
    print("DEFENDER EVOLUTION COMPLETE")
    print("=" * 60)

    best = hof[0]
    best_n_estimators = int(best[0])
    best_learning_rate = float(best[1])
    best_num_leaves = int(best[2])

    best_score, best_auc, best_robust = defender_score_and_metrics(
        best, X_tr, y_tr, X_te, y_te, attackers
    )

    print(f"\nBest hyperparameters found:")
    print(f"  n_estimators: {best_n_estimators}")
    print(f"  learning_rate: {best_learning_rate:.6f}")
    print(f"  num_leaves: {best_num_leaves}")
    print(f"  Clean AUC: {best_auc:.6f}")
    print(f"  Robust detection: {best_robust:.6f}")
    print(f"  Combined score: {best_score:.6f}")

    # Compare with a simple baseline (clean only, no attackers)
    baseline = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        verbose=-1
    )
    baseline.fit(X_tr, y_tr)
    baseline_auc = roc_auc_score(y_te, baseline.predict_proba(X_te)[:, 1])

    print(f"\nBaseline (default params, clean AUC only):")
    print(f"  n_estimators: 400")
    print(f"  learning_rate: 0.05")
    print(f"  num_leaves: 64")
    print(f"  AUC: {baseline_auc:.6f}")

    improvement = (best_auc - baseline_auc) / baseline_auc * 100
    print(f"\nImprovement in clean AUC: {improvement:+.2f}%")

    # Save a fully trained version of the best defender on full train set
    evolved_clf = LGBMClassifier(
        n_estimators=best_n_estimators,
        learning_rate=best_learning_rate,
        num_leaves=best_num_leaves,
        verbose=-1
    )
    evolved_clf.fit(X_train, y_train)

    evolved_path = ROOT_DIR / "models" / "evolved_defender.pkl"
    evolved_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(evolved_clf, evolved_path)
    print(f"\nBest defender model (trained on full data) saved to: {evolved_path}")

    # ---------------------------------------------------------
    # UPDATE GLOBAL LIST OF BEST DEFENDERS (TOP 30)
    # ---------------------------------------------------------
    # Refresh list (re-evaluated earlier if existed)
    if not evolved_defenders:
        evolved_defenders = []

    evolved_defenders.append({
        "score": best_score,
        "auc_clean": best_auc,
        "robust_detection": best_robust,
        "n_estimators": best_n_estimators,
        "learning_rate": best_learning_rate,
        "num_leaves": best_num_leaves,
    })

    evolved_defenders.sort(key=lambda d: d["score"], reverse=True)
    evolved_defenders = evolved_defenders[:30]

    save_evolved_defenders(evolved_defenders)
    print(f"\nUpdated evolved_defenders (top {len(evolved_defenders)}) "
          f"and saved to {EVOLVED_DEFENDERS_FILE}")


if __name__ == "__main__":
    main()
