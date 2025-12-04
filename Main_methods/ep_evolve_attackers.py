#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import random
from pathlib import Path
import sys
import joblib # type: ignore

import numpy as np
from deap import base, creator, tools
from lightgbm import LGBMClassifier  # only for type sanity; we don't train defenders here # type: ignore

# ----------------------------------------------------------------------
# Paths / globals
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent  # EA-in-Cyber
RESULTS_DIR = ROOT_DIR / "results-ep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVOLVED_ATTACKERS_FILE = RESULTS_DIR / "evolved_attackers.pkl"
RANDOM_DEFENDERS_DIR = ROOT_DIR / "models" / "random_defenders"

# thrember
sys.path.insert(0, str(ROOT_DIR / "General" / "Example"))
import thrember  # type: ignore

# EP hyperparameters
EPS = 0.5
BATCH_SIZE = 256
K = 50                    # number of features attacker can perturb
MU = 100                  # attacker population size
LAMBDA_PER_PARENT = 3     # offspring per parent
N_GEN = 20                # EP generations
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ----------------------------------------------------------------------
# Data + defender loading
# ----------------------------------------------------------------------
def load_ember_malware(subset_size=5000):
    data_dir = ROOT_DIR / "General" / "Example" / "data"
    print(f"Loading EMBER malware from: {data_dir}")
    X_train, y_train = thrember.read_vectorized_features(str(data_dir), subset="train")
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)

    mask = (y_train == 1)
    X_mal = X_train[mask]

    print(f"  Total malware samples: {len(X_mal)}")
    if len(X_mal) > subset_size:
        idx = np.random.choice(len(X_mal), subset_size, replace=False)
        X_mal = X_mal[idx]

    print(f"  Using {len(X_mal)} malware samples for attacker EP")
    return X_mal


def load_defender_pool():
    """
    Load all defenders from models/random_defenders/ as sklearn models.
    """
    if not RANDOM_DEFENDERS_DIR.exists():
        raise FileNotFoundError(f"Defender dir not found: {RANDOM_DEFENDERS_DIR}")

    pkls = sorted(RANDOM_DEFENDERS_DIR.glob("defender_*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No defender_*.pkl found in {RANDOM_DEFENDERS_DIR}")

    print(f"Loading {len(pkls)} defenders from: {RANDOM_DEFENDERS_DIR}")
    defenders = []
    for p in pkls:
        clf = joblib.load(p)
        defenders.append(clf)
        print(f"  loaded {p.name}")
    return defenders


# ----------------------------------------------------------------------
# Persistent attackers list
# ----------------------------------------------------------------------
def load_evolved_attackers():
    if EVOLVED_ATTACKERS_FILE.exists():
        return joblib.load(EVOLVED_ATTACKERS_FILE)
    return []


def save_evolved_attackers(evolved):
    joblib.dump(evolved, EVOLVED_ATTACKERS_FILE)


# ----------------------------------------------------------------------
# Evasion computation
# ----------------------------------------------------------------------
def compute_evasion(genes, feat_idx, X_malware, defenders):
    if len(defenders) == 0:
        return 0.0

    if len(X_malware) <= BATCH_SIZE:
        Xb = X_malware
    else:
        idx = np.random.choice(len(X_malware), BATCH_SIZE, replace=False)
        Xb = X_malware[idx]

    X_adv = Xb.copy()
    deltas = np.array(genes, dtype=float) * EPS
    X_adv[:, feat_idx] += deltas
    X_adv = np.clip(X_adv, 0.0, None)

    evasion_rates = []
    for clf in defenders:
        proba = clf.predict_proba(X_adv)[:, 1]
        benign_pred = (proba < 0.5).astype(int)
        evasion_rates.append(benign_pred.mean())

    return float(np.mean(evasion_rates))


def recompute_evasion_for_attacker(attacker, X_malware, defenders):
    genes = attacker["genes"]
    feat_idx = np.array(attacker["feature_indices"], dtype=int)
    return compute_evasion(genes, feat_idx, X_malware, defenders)


def evaluate_attacker(individual, X_malware, defenders, feat_idx):
    try:
        ev = compute_evasion(individual, feat_idx, X_malware, defenders)
        return (-ev,)  # minimize negative evasion
    except Exception as e:
        print("Error in attacker eval:", e)
        return (1.0,)


# ----------------------------------------------------------------------
# EP main
# ----------------------------------------------------------------------
def main():
    print("=== EP: evolve attackers ===")
    X_malware = load_ember_malware(subset_size=5000)
    defenders = load_defender_pool()

    # 1) load + re-score old attackers
    old_attackers = load_evolved_attackers()
    if old_attackers:
        print(f"Re-scoring {len(old_attackers)} stored attackers vs current defenders...")
        for atk in old_attackers:
            atk["evasion"] = recompute_evasion_for_attacker(atk, X_malware, defenders)
    else:
        print("No stored attackers (first EP run).")

    # 2) setup DEAP EP for new attackers
    n_features = X_malware.shape[1]
    feat_idx = np.random.choice(n_features, K, replace=False)
    print(f"Attacker perturbs {K} features: {feat_idx[:10]}...")

    try:
        creator.create("A_FitnessMin", base.Fitness, weights=(-1.0,))
    except Exception:
        pass
    try:
        creator.create("Attacker", list, fitness=creator.A_FitnessMin)
    except Exception:
        pass

    toolbox = base.Toolbox()
    toolbox.register("gene", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Attacker, toolbox.gene, K)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_wrap(ind):
        return evaluate_attacker(ind, X_malware, defenders, feat_idx)

    toolbox.register("evaluate", eval_wrap)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.3, indpb=0.3)

    # 3) initialize population
    pop = toolbox.population(n=MU)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # 4) EP loop: (μ + λ) selection, mutation-only
    for gen in range(1, N_GEN + 1):
        print(f"\n[EP attackers] Generation {gen}/{N_GEN}")

        offspring = []
        for parent in pop:
            for _ in range(LAMBDA_PER_PARENT):
                child = toolbox.clone(parent)
                toolbox.mutate(child)
                # clip genes to [-1, 1]
                for i in range(len(child)):
                    child[i] = float(np.clip(child[i], -1.0, 1.0))
                child.fitness.values = toolbox.evaluate(child)
                offspring.append(child)

        # (μ + λ) selection
        combined = pop + offspring
        pop = tools.selBest(combined, MU)

        fitness_vals = [ind.fitness.values[0] for ind in pop]
        print(f"  best = {min(fitness_vals):.4f}, avg = {np.mean(fitness_vals):.4f}")

    # 5) merge with old attackers, keep best 1000 by evasion
    print("\nMerging EP attackers with stored attackers...")
    for ind in pop:
        ev = -ind.fitness.values[0]
        old_attackers.append({
            "evasion": float(ev),
            "genes": np.array(ind, dtype=float),
            "feature_indices": feat_idx.astype(int),
        })

    old_attackers.sort(key=lambda a: a["evasion"], reverse=True)
    old_attackers = old_attackers[:1000]

    save_evolved_attackers(old_attackers)
    print(f"Saved top {len(old_attackers)} attackers to {EVOLVED_ATTACKERS_FILE}")


if __name__ == "__main__":
    main()