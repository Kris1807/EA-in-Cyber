#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import random
from pathlib import Path
import sys
import joblib # type: ignore

import numpy as np
from deap import base, creator, tools
from lightgbm import LGBMClassifier # type: ignore
from sklearn.metrics import roc_auc_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# ----------------------------------------------------------------------
# Paths / globals
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent  # EA-in-Cyber
RESULTS_DIR = ROOT_DIR / "results-ep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVOLVED_ATTACKERS_FILE = RESULTS_DIR / "evolved_attackers.pkl"
EVOLVED_DEFENDERS_FILE = RESULTS_DIR / "evolved_defenders.pkl"
RANDOM_DEFENDERS_DIR = ROOT_DIR / "models" / "random_defenders"

# thrember
sys.path.insert(0, str(ROOT_DIR / "General" / "Example"))
import thrember  # type: ignore

# EP hyperparameters
EPS = 0.5
ADV_BATCH_SIZE = 256

MU = 30                 # defender population size
LAMBDA_PER_PARENT = 3   # offspring per parent
N_GEN = 20
RANDOM_SEED = 123

# bounds for defender hyperparameters
DEF_MIN = [100.0, 0.001, 10.0]
DEF_MAX = [1000.0, 0.1, 255.0]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_ember_data(subset="train", subset_size=5000):
    data_dir = ROOT_DIR / "General" / "Example" / "data"
    print(f"Loading EMBER data from: {data_dir} (subset='{subset}')")
    X, y = thrember.read_vectorized_features(str(data_dir), subset=subset)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if len(X) > subset_size:
        idx = np.random.choice(len(X), subset_size, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y


# ----------------------------------------------------------------------
# Attackers and defenders persistent lists
# ----------------------------------------------------------------------
def load_top_attackers(max_attackers=20):
    if not EVOLVED_ATTACKERS_FILE.exists():
        print("No evolved attackers yet. Defenders will be scored only by clean AUC.")
        return []
    attackers = joblib.load(EVOLVED_ATTACKERS_FILE)
    attackers.sort(key=lambda a: a.get("evasion", 0.0), reverse=True)
    return attackers[:max_attackers]


def load_evolved_defenders():
    if EVOLVED_DEFENDERS_FILE.exists():
        return joblib.load(EVOLVED_DEFENDERS_FILE)
    return []


def save_evolved_defenders(defenders):
    joblib.dump(defenders, EVOLVED_DEFENDERS_FILE)


def load_initial_defender_hparams():
    """
    Seed hyperparameters from models/random_defenders/ if no evolved defenders.
    """
    if not RANDOM_DEFENDERS_DIR.exists():
        raise FileNotFoundError(f"Defender dir not found: {RANDOM_DEFENDERS_DIR}")

    pkls = sorted(RANDOM_DEFENDERS_DIR.glob("defender_*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No defender_*.pkl found in {RANDOM_DEFENDERS_DIR}")

    print(f"Loading initial defenders from {RANDOM_DEFENDERS_DIR}")
    individuals = []
    for p in pkls:
        clf = joblib.load(p)
        params = clf.get_params()
        n_estimators = float(params.get("n_estimators", 400))
        learning_rate = float(params.get("learning_rate", 0.05))
        num_leaves = float(params.get("num_leaves", 64))
        individuals.append([n_estimators, learning_rate, num_leaves])
        print(f"  {p.name}: n_estimators={n_estimators}, lr={learning_rate}, num_leaves={num_leaves}")
    return individuals


# ----------------------------------------------------------------------
# Defender scoring vs attackers
# ----------------------------------------------------------------------
def defender_score_and_metrics(hyperparams, X_tr, y_tr, X_te, y_te, attackers):
    n_estimators = max(1, int(round(hyperparams[0])))
    learning_rate = float(hyperparams[1])
    num_leaves = max(2, int(round(hyperparams[2])))

    clf = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        verbose=-1
    )
    clf.fit(X_tr, y_tr)

    proba_clean = clf.predict_proba(X_te)[:, 1]
    auc_clean = float(roc_auc_score(y_te, proba_clean))

    if not attackers:
        return auc_clean, auc_clean, 0.0

    # build malware subset from test
    mal_mask = (y_te == 1)
    X_mal = X_te[mal_mask]
    if len(X_mal) == 0:
        robust_det = 0.0
    else:
        if len(X_mal) > ADV_BATCH_SIZE:
            idx = np.random.choice(len(X_mal), ADV_BATCH_SIZE, replace=False)
            Xb = X_mal[idx]
        else:
            Xb = X_mal

        det_rates = []
        for atk in attackers:
            genes = np.array(atk["genes"], dtype=float)
            feat_idx = np.array(atk["feature_indices"], dtype=int)
            if genes.size == 0 or feat_idx.size == 0:
                continue
            if genes.size != feat_idx.size:
                m = min(genes.size, feat_idx.size)
                genes = genes[:m]
                feat_idx = feat_idx[:m]

            X_adv = Xb.copy()
            X_adv[:, feat_idx] += genes * EPS
            X_adv = np.clip(X_adv, 0.0, None)

            proba_adv = clf.predict_proba(X_adv)[:, 1]
            benign_pred = (proba_adv < 0.5).astype(int)
            det_rates.append(1.0 - benign_pred.mean())

        robust_det = float(np.mean(det_rates)) if det_rates else 0.0

    score = 0.5 * auc_clean + 0.5 * robust_det
    return score, auc_clean, robust_det


def evaluate_defender(individual, X_tr, y_tr, X_te, y_te, attackers):
    try:
        score, _, _ = defender_score_and_metrics(individual, X_tr, y_tr, X_te, y_te, attackers)
        return (-score,)   # minimize negative score
    except Exception as e:
        print("Error in defender eval:", e)
        return (1.0,)


# ----------------------------------------------------------------------
# EP main
# ----------------------------------------------------------------------
def main():
    print("=== EP: evolve defenders ===")

    # load data
    X, y = load_ember_data(subset="train", subset_size=5000)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train size: {len(X_tr)}, Test size: {len(X_te)}")

    # load current attackers
    attackers = load_top_attackers(max_attackers=20)

    # load evolved defenders list (if any) and re-score them
    evolved_defenders = load_evolved_defenders()
    if evolved_defenders:
        print(f"Re-evaluating {len(evolved_defenders)} stored defenders...")
        for d in evolved_defenders:
            hyper = [d["n_estimators"], d["learning_rate"], d["num_leaves"]]
            score, auc_clean, robust_det = defender_score_and_metrics(
                hyper, X_tr, y_tr, X_te, y_te, attackers
            )
            d["score"] = score
            d["auc_clean"] = auc_clean
            d["robust_detection"] = robust_det
    else:
        print("No stored defenders yet; seeding from random_defenders dir.")

    # DEAP EP setup
    try:
        creator.create("D_FitnessMin", base.Fitness, weights=(-1.0,))
    except Exception:
        pass
    try:
        creator.create("Defender", list, fitness=creator.D_FitnessMin)
    except Exception:
        pass

    toolbox = base.Toolbox()
    toolbox.register("n_estimators_attr", random.uniform, DEF_MIN[0], DEF_MAX[0])
    toolbox.register("lr_attr", random.uniform, DEF_MIN[1], DEF_MAX[1])
    toolbox.register("num_leaves_attr", random.uniform, DEF_MIN[2], DEF_MAX[2])
    toolbox.register(
        "individual_random",
        tools.initCycle,
        creator.Defender,
        (toolbox.n_estimators_attr, toolbox.lr_attr, toolbox.num_leaves_attr),
        n=1
    )

    def init_from_list(vals):
        return creator.Defender(vals)
    toolbox.register("individual_from_list", init_from_list)

    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.3)

    def eval_wrap(ind):
        # clip to bounds before evaluation
        for i, (mn, mx) in enumerate(zip(DEF_MIN, DEF_MAX)):
            ind[i] = float(np.clip(ind[i], mn, mx))
        return evaluate_defender(ind, X_tr, y_tr, X_te, y_te, attackers)

    toolbox.register("evaluate", eval_wrap)

    # seed population
    if evolved_defenders:
        initial_hparams = [
            [float(d["n_estimators"]), float(d["learning_rate"]), float(d["num_leaves"])]
            for d in evolved_defenders
        ]
    else:
        initial_hparams = load_initial_defender_hparams()

    # if you have more than MU, just take first MU
    if len(initial_hparams) > MU:
        initial_hparams = initial_hparams[:MU]

    pop = [toolbox.individual_from_list(h) for h in initial_hparams]
    # if fewer than MU, fill the rest randomly
    while len(pop) < MU:
        pop.append(toolbox.individual_random())

    # initial eval
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # EP loop
    for gen in range(1, N_GEN + 1):
        print(f"\n[EP defenders] Generation {gen}/{N_GEN}")

        offspring = []
        for parent in pop:
            for _ in range(LAMBDA_PER_PARENT):
                child = toolbox.clone(parent)
                toolbox.mutate(child)
                for i, (mn, mx) in enumerate(zip(DEF_MIN, DEF_MAX)):
                    child[i] = float(np.clip(child[i], mn, mx))
                child.fitness.values = toolbox.evaluate(child)
                offspring.append(child)

        combined = pop + offspring
        pop = tools.selBest(combined, MU)

        fitness_vals = [ind.fitness.values[0] for ind in pop]
        print(f"  best = {min(fitness_vals):.4f}, avg = {np.mean(fitness_vals):.4f}")

    # Merge into persistent top defenders (top 30)
    print("\nMerging EP defenders with stored defenders...")
    if evolved_defenders is None:
        evolved_defenders = []

    for ind in pop:
        score, auc, robust = defender_score_and_metrics(ind, X_tr, y_tr, X_te, y_te, attackers)
        evolved_defenders.append({
            "score": score,
            "auc_clean": auc,
            "robust_detection": robust,
            "n_estimators": int(round(ind[0])),
            "learning_rate": float(ind[1]),
            "num_leaves": int(round(ind[2])),
        })

    evolved_defenders.sort(key=lambda d: d["score"], reverse=True)
    evolved_defenders = evolved_defenders[:30]
    save_evolved_defenders(evolved_defenders)
    print(f"Saved top {len(evolved_defenders)} defenders to {EVOLVED_DEFENDERS_FILE}")

    # Optionally, also save the single best defender as a trained model on full X,y
    best = evolved_defenders[0]
    print("\nBest defender after EP:")
    print(best)

    best_clf = LGBMClassifier(
        n_estimators=int(best["n_estimators"]),
        learning_rate=float(best["learning_rate"]),
        num_leaves=int(best["num_leaves"]),
        verbose=-1
    )
    best_clf.fit(X, y)
    out = ROOT_DIR / "models" / "ep_best_defender.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, out)
    print(f"Saved ep_best_defender model to: {out}")


if __name__ == "__main__":
    main()
