import warnings

#Only way to get rid of the warnings 
#The warnings are seemingly harmless but they're annoying
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import numpy as np
import random
from deap import base, creator, tools, algorithms
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "General" / "Example"))
import thrember


def load_ember_data(data_path="General/Example/data", subset="train", subset_size=5000):
    
    #load ember data as before
    try:
        X, y = thrember.read_vectorized_features(data_path, subset=subset)

        #subset so it's quicker...
        if len(X) > subset_size:
            indices = np.random.choice(len(X), subset_size, replace=False)
            X = X[indices]
            y = y[indices]

        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def evaluate_defender(hyperparams, X_train, y_train, X_test, y_test):

    #train a defender with given hyperparameters and evaluate
    try:
        n_estimators = int(hyperparams[0])
        learning_rate = hyperparams[1]
        num_leaves = max(2, int(hyperparams[2]))

        #train model
        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            verbose=-1
        )
        model.fit(X_train, y_train)

        #evaluate on test set
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

        # return negative auc bc DEAP minimizes by default
        return (-auc,)
    except Exception as e:
        print(f"Error in eval: {e}")
        return (1.0,) 


def main():
    print("Loading EMBER2024 data...")
    X_train, y_train = load_ember_data(subset_size=5000)


    print(f"Loaded {len(X_train)} samples with {X_train.shape[1]} features")

    #small validation split from train
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

    #setup DEAP!!
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    #hyperparameter bounds
    toolbox.register("n_estimators", random.uniform, 100, 1000)
    toolbox.register("learning_rate", random.uniform, 0.001, 0.1)
    toolbox.register("num_leaves", random.uniform, 10, 255)

    # individual looks like: [n_estimators, learning_rate, num_leaves]
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.n_estimators, toolbox.learning_rate, toolbox.num_leaves),
                     n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #genetic operators (for now im not using the ones you guys made)
    toolbox.register("evaluate", lambda ind: evaluate_defender(ind, X_tr, y_tr, X_te, y_te))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    #bounds for mutation
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

    #run the algo
    print("\nStarting genetic algorithm...")
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.7,  # crossover probability
        mutpb=0.3,  # mutation probability
        ngen=10,  # number of generations (this is temporarily smaller to be faster) 
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    #results
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)

    best = hof[0]
    best_auc = -best.fitness.values[0]

    print(f"\nBest hyperparameters found:")
    print(f"  n_estimators: {int(best[0])}")
    print(f"  learning_rate: {best[1]:.6f}")
    print(f"  num_leaves: {int(best[2])}")
    print(f"  AUC: {best_auc:.6f}")

    # compate with baseline (what it would be if we didn't evolve)
    baseline = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        verbose=-1
    )
    baseline.fit(X_tr, y_tr)
    baseline_auc = roc_auc_score(y_te, baseline.predict_proba(X_te)[:, 1])

    print(f"\nBaseline (default params):")
    print(f"  n_estimators: 400")
    print(f"  learning_rate: 0.05")
    print(f"  num_leaves: 64")
    print(f"  AUC: {baseline_auc:.6f}")

    improvement = (best_auc - baseline_auc) / baseline_auc * 100
    print(f"\nImprovement: {improvement:+.2f}%")


if __name__ == "__main__":
    main()
