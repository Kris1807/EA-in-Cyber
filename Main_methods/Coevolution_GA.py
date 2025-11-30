#!/usr/bin/env python3
"""
Coevolutionary attacker-defender GA.

Combines your attacker & defender logic to co-evolve both populations.
"""
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import os
from pathlib import Path
import sys
import random
import joblib
import numpy as np

# ML
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# DEAP
from deap import base, creator, tools, algorithms

# ----------------------------------------------------------------------
# Paths / globals (adapt to your repo structure)
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "General" / "Example" / "data"
MODELS_DIR = ROOT_DIR / "models" / "defenders"   # for seed defenders if present
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVOLVED_ATTACKERS_FILE = RESULTS_DIR / "evolved_attackers.pkl"
EVOLVED_DEFENDERS_FILE = RESULTS_DIR / "evolved_defenders.pkl"

# Bring your thrember into path if present (same pattern as original)
sys.path.insert(0, str(ROOT_DIR / "General" / "Example"))
try:
    import thrember
except Exception as e:
    print("Warning: failed to import thrember:", e)
    thrember = None

# ----------------------------------------------------------------------
# Coevolution hyperparameters (tweak these)
# ----------------------------------------------------------------------
EPS = 0.5                     # attacker perturbation scale
ADV_BATCH_SIZE = 256          # adversarial batch size used during evaluations
K = 50                        # number of features attacker can perturb (genes length)
ATT_POP = 75                  # attacker population
DEF_POP = 20                  # defender population
NGEN = 20                     # number of coevolution generations
ATT_CX = 0.7
ATT_MUT = 0.3
DEF_CX = 0.8
DEF_MUT = 0.5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------------
# Utility: load EMBER data (uses thrember like your original code)
# ----------------------------------------------------------------------
def load_ember(subset="train", subset_size=5000):
    print(f"Loading EMBER data from: {DATA_DIR} (subset={subset})")
    if thrember is None:
        print("thrember not available. Please ensure thrember is importable.")
        return None, None
    try:
        X, y = thrember.read_vectorized_features(str(DATA_DIR), subset=subset)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if subset_size is not None and len(X) > subset_size:
            sel = np.random.choice(len(X), subset_size, replace=False)
            X = X[sel]
            y = y[sel]
        return X, y
    except Exception as e:
        print(f"Failed to load EMBER dataset: {e}")
        return None, None

# ----------------------------------------------------------------------
# Attacker evaluation (against a list of defender classifiers)
# attacker individual: list of length K with floats in [-1,1]
# ----------------------------------------------------------------------
def evaluate_attacker_ind(individual, X_malware, defenders, feature_idx):
    """
    Return 1-tuple: (-avg_evasion,) because DEAP minimizes.
    """
    try:
        if len(defenders) == 0:
            # No defenders -> assign poor fitness (high positive) so GA will explore
            return (1.0,)

        # sample malware batch
        idx = np.random.choice(len(X_malware), min(ADV_BATCH_SIZE, len(X_malware)), replace=False)
        Xb = X_malware[idx].astype(float).copy()

        deltas = np.array(individual, dtype=float) * EPS
        # apply to feature columns
        Xb[:, feature_idx] += deltas
        Xb = np.clip(Xb, 0.0, None)

        evasion_rates = []
        for clf in defenders:
            try:
                proba = clf.predict_proba(Xb)[:, 1]
                evasion = (proba < 0.5).mean()
                evasion_rates.append(evasion)
            except Exception:
                # if defender fails to predict (bad model), skip
                continue

        if not evasion_rates:
            return (1.0,)
        avg_evasion = float(np.mean(evasion_rates))
        return (-avg_evasion,)
    except Exception as e:
        print("Error in attacker eval:", e)
        return (1.0,)

# ----------------------------------------------------------------------
# Defender evaluation (against attackers)
# defender individual: [n_estimators, learning_rate, num_leaves]
# ----------------------------------------------------------------------
def defender_score_and_metrics(hyperparams, X_train, y_train, X_test, y_test, attackers):
    """
    Returns (score, auc_clean, robust_detection)
    """
    n_estimators = max(1, int(round(hyperparams[0])))
    learning_rate = float(hyperparams[1])
    num_leaves = max(2, int(round(hyperparams[2])))

    clf = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        verbose=-1
    )
    clf.fit(X_train, y_train)

    proba_clean = clf.predict_proba(X_test)[:, 1]
    auc_clean = float(roc_auc_score(y_test, proba_clean))

    if not attackers:
        return auc_clean, auc_clean, 0.0

    # build malware batch from test set
    mal_mask = (y_test == 1)
    X_mal = X_test[mal_mask]
    if len(X_mal) == 0:
        robust_det = 0.0
    else:
        if len(X_mal) > ADV_BATCH_SIZE:
            idx = np.random.choice(len(X_mal), ADV_BATCH_SIZE, replace=False)
            X_batch = X_mal[idx]
        else:
            X_batch = X_mal

        det_rates = []
        for atk in attackers:
            genes = np.array(atk.get("genes", []), dtype=float)
            feat_idx = np.array(atk.get("feature_indices", []), dtype=int)
            if genes.size == 0 or feat_idx.size == 0:
                continue
            # align shapes
            if genes.size != feat_idx.size:
                min_len = min(genes.size, feat_idx.size)
                genes = genes[:min_len]
                feat_idx = feat_idx[:min_len]

            X_adv = X_batch.copy()
            X_adv[:, feat_idx] += genes * EPS
            X_adv = np.clip(X_adv, 0.0, None)

            proba_adv = clf.predict_proba(X_adv)[:, 1]
            benign_pred = (proba_adv < 0.5).astype(int)
            detection_rate = 1.0 - benign_pred.mean()
            det_rates.append(detection_rate)

        robust_det = float(np.mean(det_rates)) if det_rates else 0.0

    score = 0.5 * auc_clean + 0.5 * robust_det
    return score, auc_clean, robust_det

def evaluate_defender_ind(individual, X_train, y_train, X_test, y_test, attackers):
    """
    DEAP wrapper: returns (-score,)
    """
    try:
        score, _, _ = defender_score_and_metrics(individual, X_train, y_train, X_test, y_test, attackers)
        return (-score,)
    except Exception as e:
        print("Error in defender eval:", e)
        return (1.0,)

# ----------------------------------------------------------------------
# Helpers to persist top individuals as simple dicts for cross-eval
# ----------------------------------------------------------------------
def attacker_to_dict(ind, feature_idx, evasion=None):
    return {"genes": np.array(ind, dtype=float), "feature_indices": np.array(feature_idx, dtype=int), "evasion": float(evasion) if evasion is not None else None}

def defender_to_dict(ind, score=None, auc=None, robust=None):
    return {"n_estimators": int(round(ind[0])), "learning_rate": float(ind[1]), "num_leaves": int(round(ind[2])),
            "score": score, "auc_clean": auc, "robust_detection": robust}

# ----------------------------------------------------------------------
# Main coevolutionary GA
# ----------------------------------------------------------------------
def coevolve_main():
    # Load data
    X, y = load_ember(subset="train", subset_size=5000)
    if X is None:
        print("Failed to load data. Exiting.")
        return

    # small train/test split (defender uses this)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    # prepare malware pool for attacker evaluation (use train set malware)
    mal_mask = (y_tr == 1)
    X_malware = X_tr[mal_mask]
    if len(X_malware) == 0:
        print("No malware samples in training set. Exiting.")
        return

    # pick global feature indices that attackers will perturb
    FEATURE_IDX = np.random.choice(X.shape[1], K, replace=False)
    print(f"Using {K} feature indices for attackers: {FEATURE_IDX[:10]} ...")

    # DEAP setup for attacker
    try:
        creator.create("A_FitnessMin", base.Fitness, weights=(-1.0,))
    except Exception:
        pass
    try:
        creator.create("Attacker", list, fitness=creator.A_FitnessMin)
    except Exception:
        pass

    att_toolbox = base.Toolbox()
    att_toolbox.register("gene", random.uniform, -1.0, 1.0)
    att_toolbox.register("individual", tools.initRepeat, creator.Attacker, att_toolbox.gene, K)
    att_toolbox.register("population", tools.initRepeat, list, att_toolbox.individual)

    # DEAP setup for defender
    try:
        creator.create("D_FitnessMin", base.Fitness, weights=(-1.0,))
    except Exception:
        pass
    try:
        creator.create("Defender", list, fitness=creator.D_FitnessMin)
    except Exception:
        pass

    def_toolbox = base.Toolbox()
    # bounds for defender hyperparams
    DEF_MIN = [100.0, 0.001, 10.0]
    DEF_MAX = [1000.0, 0.1, 255.0]
    def_toolbox.register("n_estimators_attr", random.uniform, DEF_MIN[0], DEF_MAX[0])
    def_toolbox.register("learning_rate_attr", random.uniform, DEF_MIN[1], DEF_MAX[1])
    def_toolbox.register("num_leaves_attr", random.uniform, DEF_MIN[2], DEF_MAX[2])
    def_toolbox.register("individual", tools.initCycle, creator.Defender,
                         (def_toolbox.n_estimators_attr, def_toolbox.learning_rate_attr, def_toolbox.num_leaves_attr), n=1)
    def_toolbox.register("population", tools.initRepeat, list, def_toolbox.individual)

    # Register genetic operators
    # Attackers
    att_toolbox.register("mate", tools.cxBlend, alpha=0.5)
    att_toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
    att_toolbox.register("select", tools.selTournament, tournsize=3)

    # Defenders
    def_toolbox.register("mate", tools.cxBlend, alpha=0.5)
    def_toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    def_toolbox.register("select", tools.selTournament, tournsize=3)

    # bounds decorator for defenders
    def check_bounds_decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            # ensure list/tuple handling
            res = []
            for ind in offspring:
                for i, (mn, mx) in enumerate(zip(DEF_MIN, DEF_MAX)):
                    if ind[i] < mn: ind[i] = mn
                    if ind[i] > mx: ind[i] = mx
                res.append(ind)
            return type(offspring)(res)
        return wrapper

    def_toolbox.decorate("mate", check_bounds_decorator)
    def_toolbox.decorate("mutate", check_bounds_decorator)

    # init populations
    pop_att = att_toolbox.population(n=ATT_POP)
    pop_def = def_toolbox.population(n=DEF_POP)

    # Hall of fame & stats
    hof_att = tools.HallOfFame(5)
    hof_def = tools.HallOfFame(5)

    att_stats = tools.Statistics(lambda ind: ind.fitness.values)
    att_stats.register("avg", np.mean)
    att_stats.register("min", np.min)
    att_stats.register("max", np.max)

    def_stats = tools.Statistics(lambda ind: ind.fitness.values)
    def_stats.register("avg", np.mean)
    def_stats.register("min", np.min)
    def_stats.register("max", np.max)

    # initial evaluation: defenders are untrained; evaluate defenders on current attackers (empty -> use clean AUC)
    # but we need initial attackers’ fitness first; attackers evaluated against current defenders (initial random defenders).
    # We'll do evaluations inside the main loop.

    print("\nStarting co-evolution")
    for gen in range(1, NGEN+1):
        print(f"\n===== Generation {gen} =====")

        # ------- Evaluate attacker population against current defenders -------
        # turn defenders (pop_def) into real sklearn classifiers for attacker eval
        defenders_clfs = []
        for d in pop_def:
            try:
                n_est = int(round(d[0])); lr = float(d[1]); nl = int(round(d[2]))
                clf = LGBMClassifier(n_estimators=n_est, learning_rate=lr, num_leaves=nl, verbose=-1)
                # train on X_tr/y_tr quickly (defender evaluation uses small models)
                clf.fit(X_tr, y_tr)
                defenders_clfs.append(clf)
            except Exception:
                continue

        # Evaluate attackers
        for ind in pop_att:
            ind.fitness.values = evaluate_attacker_ind(ind, X_malware, defenders_clfs, FEATURE_IDX)

        # record HOF, stats
        hof_att.update(pop_att)
        att_summary = att_stats.compile(pop_att)
        print(f"Attackers stats: {att_summary}")

        # Evolve attackers one generation (selection, mating, mutation)
        offspring_att = att_toolbox.select(pop_att, len(pop_att))
        offspring_att = list(map(att_toolbox.clone, offspring_att))

        # crossover
        for i in range(1, len(offspring_att), 2):
            if random.random() < ATT_CX:
                att_toolbox.mate(offspring_att[i-1], offspring_att[i])
                # ensure bounds
                for j in (i-1, i):
                    for k in range(len(offspring_att[j])):
                        offspring_att[j][k] = float(np.clip(offspring_att[j][k], -1.0, 1.0))

        # mutation
        for mutant in offspring_att:
            if random.random() < ATT_MUT:
                att_toolbox.mutate(mutant)
                for k in range(len(mutant)):
                    mutant[k] = float(np.clip(mutant[k], -1.0, 1.0))

        pop_att = offspring_att

        # ------- Build attacker archive (top N) to evaluate defenders -------
        # Convert top attackers into dicts with genes & feature_indices for defender eval
        top_attackers = []
        # evaluate all attackers against defenders_clfs to get evasion numbers for sorting
        attacker_eval_records = []
        for ind in pop_att:
            ev = evaluate_attacker_ind(ind, X_malware, defenders_clfs, FEATURE_IDX)[0]
            attacker_eval_records.append((ind, -ev))  # ev was negative avg_evasion in returned tuple
        # sort by evasion desc
        attacker_eval_records.sort(key=lambda t: t[1], reverse=True)
        # take top 20 or all if fewer
        top_count = min(20, len(attacker_eval_records))
        for i in range(top_count):
            ind, evasion = attacker_eval_records[i]
            top_attackers.append(attacker_to_dict(ind, FEATURE_IDX, evasion=float(evasion)))

        # ------- Evaluate defender population against current attackers -------
        # Now evaluate each defender (hyperparams) using evaluate_defender_ind wrapper
        # pass the attacker archive (top_attackers) into the evaluation
        for d in pop_def:
            d.fitness.values = evaluate_defender_ind(d, X_tr, y_tr, X_te, y_te, top_attackers)

        hof_def.update(pop_def)
        def_summary = def_stats.compile(pop_def)
        print(f"Defenders stats: {def_summary}")

        # Evolve defenders one generation
        offspring_def = def_toolbox.select(pop_def, len(pop_def))
        offspring_def = list(map(def_toolbox.clone, offspring_def))

        # crossover
        for i in range(1, len(offspring_def), 2):
            if random.random() < DEF_CX:
                def_toolbox.mate(offspring_def[i-1], offspring_def[i])

        # mutation
        for mutant in offspring_def:
            if random.random() < DEF_MUT:
                def_toolbox.mutate(mutant)

        # enforce bounds (already done via decorator but ensure)
        for ind in offspring_def:
            for i, (mn, mx) in enumerate(zip(DEF_MIN, DEF_MAX)):
                ind[i] = float(np.clip(ind[i], mn, mx))

        pop_def = offspring_def

        # Optionally: every N gens save current HOF to disk
        if gen % 5 == 0 or gen == NGEN:
            # save top attackers and defenders
            saved_attackers = [attacker_to_dict(h, FEATURE_IDX, evasion=None) for h in hof_att]
            joblib.dump(saved_attackers, EVOLVED_ATTACKERS_FILE)
            saved_defenders = []
            for h in hof_def:
                # re-evaluate for reporting
                score, auc, robust = defender_score_and_metrics(h, X_tr, y_tr, X_te, y_te, top_attackers)
                saved_defenders.append(defender_to_dict(h, score=score, auc=auc, robust=robust))
            joblib.dump(saved_defenders, EVOLVED_DEFENDERS_FILE)
            print(f"[Saved] Top attackers -> {EVOLVED_ATTACKERS_FILE}; Top defenders -> {EVOLVED_DEFENDERS_FILE}")

    # End of coevolution
    print("\n=== COEVOLUTION COMPLETE ===")
    # final bests
    if len(hof_att) > 0:
        best_att = hof_att[0]
        ev = evaluate_attacker_ind(best_att, X_malware, defenders_clfs, FEATURE_IDX)[0]
        print(f"Best attacker (genes first 10): {list(best_att)[:10]}  estimated evasion: {-ev:.4f}")
    if len(hof_def) > 0:
        best_def = hof_def[0]
        score, auc, robust = defender_score_and_metrics(best_def, X_tr, y_tr, X_te, y_te, top_attackers)
        print(f"Best defender hyperparams: n_estimators={int(round(best_def[0]))}, lr={best_def[1]:.6f}, num_leaves={int(round(best_def[2]))}")
        print(f"  combined score={score:.6f}, auc={auc:.6f}, robust_detection={robust:.6f}")

    # Save final best defender model trained on full data
    if len(hof_def) > 0:
        best_def = hof_def[0]
        be_n = int(round(best_def[0])); be_lr = float(best_def[1]); be_nl = int(round(best_def[2]))
        final_clf = LGBMClassifier(n_estimators=be_n, learning_rate=be_lr, num_leaves=be_nl, verbose=-1)
        final_clf.fit(X, y)
        out = ROOT_DIR / "models" / "coevolved_best_defender.pkl"
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_clf, out)
        print(f"Saved best defender (trained on full data) to: {out}")

if __name__ == "__main__":
    coevolve_main()
