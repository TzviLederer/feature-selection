import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import warnings
from tqdm.auto import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)


def calculate_error_rate(X, y, kf_n_splits=10, knn_n_neighbors=5):
    kf = KFold(n_splits=kf_n_splits, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors)
    accuracies = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    return (1 - accuracies).mean()


def one_wolf_fitness(X, y, wolf, alpha):
    error_rate = calculate_error_rate(X[:, wolf.astype(bool)], y)
    return alpha * error_rate + (1 - alpha) * wolf.mean()


def two_phase_mutation(X, y, wolf, fitness, alpha, mutation_prob):
    wolf_mutated = wolf.copy()
    one_positions = np.argwhere(wolf == 1).T[0]
    for i in one_positions:
        r = np.random.rand()
        if r < mutation_prob:
            wolf_mutated[i] = 0
            mutated_fitness = one_wolf_fitness(X, y, wolf_mutated, alpha)
            if mutated_fitness < fitness:
                fitness = mutated_fitness
                wolf = wolf_mutated

    wolf_mutated = wolf.copy()
    zero_positions = np.argwhere(wolf == 0).T[0]
    for i in zero_positions:
        r = np.random.rand()
        if r < mutation_prob:
            wolf_mutated[i] = 1
            mutated_fitness = one_wolf_fitness(X, y, wolf_mutated, alpha)
            if mutated_fitness < fitness:
                fitness = mutated_fitness
                wolf = wolf_mutated

    return wolf


def calculate_fitnesses(X, y, wolfs, alpha, two_phase_mutation_prob=None):
    fitnesses = [one_wolf_fitness(X, y, wolf, alpha) for wolf in wolfs]

    if two_phase_mutation_prob:
        alpha_idx = np.argmax(fitnesses)
        new_x_alpha = two_phase_mutation(X, y, wolfs[alpha_idx], fitnesses[alpha_idx], alpha, two_phase_mutation_prob)
        wolfs[alpha_idx], fitnesses[alpha_idx] = new_x_alpha, one_wolf_fitness(X, y, new_x_alpha, alpha)

    return fitnesses


def grey_wolf_fs(X, y, n_agents=10, iterations=100, alpha=0.999, two_phase_mutation_prob=0.5):
    n = X.shape[1]
    wolfs = (np.random.rand(n_agents, n) > .5).astype(float)

    fitnesses = calculate_fitnesses(X, y, wolfs, alpha)
    sorted_index = np.argsort(fitnesses)
    x_alpha, x_beta, x_delta = [wolfs[i] for i in sorted_index[:3]]

    min_fitness = 1
    min_fitness_x_alpha = -1

    for t in tqdm(range(iterations)):
        x_abd = np.stack([x_alpha, x_beta, x_delta]).copy()

        a = 2 - t * 2 / iterations
        A = 2 * a * np.random.rand(3, n)
        C = 2 * np.random.rand(3, n)

        for wolf_ind in sorted_index:
            wolf = wolfs[wolf_ind]
            D = C * x_abd - wolf

            X_123 = x_abd - A * D
            wolfs[wolf_ind] = X_123.mean(axis=0)

        x_si = 1 / (1 + np.exp(-wolfs))
        x_binary = np.random.rand(*wolfs.shape) >= x_si
        wolfs[:] = x_binary

        fitnesses = calculate_fitnesses(X, y, wolfs, alpha, two_phase_mutation_prob=two_phase_mutation_prob)
        sorted_index = np.argsort(fitnesses)
        x_alpha, x_beta, x_delta = [wolfs[i] for i in sorted_index[:3]]

        if min(fitnesses) < min_fitness:
            min_fitness = min(fitnesses)
            min_fitness_x_alpha = x_alpha.copy()

    return min_fitness_x_alpha


def grey_wolf_fs_New(X, y, n_agents=10, iterations=100, alpha=0.999, two_phase_mutation_prob=0.5, n_layers=5):
    n = X.shape[1]
    wolfs = (np.random.rand(n_agents, n) > .5).astype(float)

    fitnesses = calculate_fitnesses(X, y, wolfs, alpha)
    sorted_index = np.argsort(fitnesses)
    x_bests_wolves = [wolfs[i] for i in sorted_index[:n_layers]]

    min_fitness = 1
    min_fitness_x_alpha = -1

    for t in range(iterations):
        x_abd = np.stack(x_bests_wolves).copy()

        a = 2 - t * 2 / iterations
        A = 2 * a * np.random.rand(n_layers, n)
        C = 2 * np.random.rand(n_layers, n)

        for wolf_ind in sorted_index:
            wolf = wolfs[wolf_ind]
            D = C * x_abd - wolf
            X_123 = x_abd - A * D
            wolfs[wolf_ind] = X_123.mean(axis=0)

        x_si = 1 / (1 + np.exp(-wolfs))
        x_binary = np.random.rand(*wolfs.shape) >= x_si
        wolfs[:] = x_binary

        fitnesses = calculate_fitnesses(X, y, wolfs, alpha, two_phase_mutation_prob=two_phase_mutation_prob)
        sorted_index = np.argsort(fitnesses)
        x_bests_wolves = [wolfs[i] for i in sorted_index[:n_layers]]

        if min(fitnesses) < min_fitness:
            min_fitness = min(fitnesses)
            min_fitness_x_alpha = x_bests_wolves[0].copy()

    return min_fitness_x_alpha
