import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import warnings
from tqdm.auto import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)


def calculate_error_rate(X, y, kf_n_splits=10, knn_n_neighbors=5):
    # calculate the error rate of wolf position according to the article
    # using 10-fold cross validation over 5-NN classifier with the wolf positions as feature selection vector
    kf = KFold(n_splits=kf_n_splits, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors)
    accuracies = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    return (1 - accuracies).mean()


def one_wolf_fitness(X, y, wolf, alpha):
    # calculating fitness of wolf according to the article - weighted sum of its error rate and its filtering rate
    error_rate = calculate_error_rate(X[:, wolf.astype(bool)], y)
    return alpha * error_rate + (1 - alpha) * wolf.mean()


def two_phase_mutation(X, y, wolf, fitness, alpha, mutation_prob):
    # apply the two phase mutation logic over the given alpha wolf

    # first phase: check each of the chosen features (with probability) whether it contributes to fitness
    # and if it does not, remove it.
    one_positions = np.argwhere(wolf == 1).T[0]
    for i in one_positions:
        r = np.random.rand()
        if r < mutation_prob:
            wolf_mutated = wolf.copy()
            wolf_mutated[i] = 0
            mutated_fitness = one_wolf_fitness(X, y, wolf_mutated, alpha)
            if mutated_fitness < fitness:
                fitness = mutated_fitness
                wolf = wolf_mutated

    # second phase: check each of the removed features (with probability) whether it contributes to fitness
    # and if it does, re-choose it.
    zero_positions = np.argwhere(wolf == 0).T[0]
    for i in zero_positions:
        r = np.random.rand()
        if r < mutation_prob:
            wolf_mutated = wolf.copy()
            wolf_mutated[i] = 1
            mutated_fitness = one_wolf_fitness(X, y, wolf_mutated, alpha)
            if mutated_fitness < fitness:
                fitness = mutated_fitness
                wolf = wolf_mutated

    return wolf


def calculate_fitnesses(X, y, wolfs, alpha, two_phase_mutation_prob=None):
    # calculate fitnesses of all wolfs according to TMGWO:

    # first calculate fitnesses for the whole pack:
    fitnesses = [one_wolf_fitness(X, y, wolf, alpha) for wolf in wolfs]

    # apply two phase mutation on the alpha wolf:
    if two_phase_mutation_prob:
        alpha_idx = np.argmax(fitnesses)
        new_x_alpha = two_phase_mutation(X, y, wolfs[alpha_idx], fitnesses[alpha_idx], alpha, two_phase_mutation_prob)
        wolfs[alpha_idx], fitnesses[alpha_idx] = new_x_alpha, one_wolf_fitness(X, y, new_x_alpha, alpha)

    return fitnesses


def grey_wolf_fs(X, y, n_agents=5, iterations=30, alpha=0.01, two_phase_mutation_prob=0.1):
    # implementation of TMGWO score function.
    # the article mentions that this process is very time-consuming (which is correct), therefore mutations are applied
    # with probability. It suggests 0.5 for best results, and 0.1 for fit time concerns.
    # Finally, we chose to use 0.1 as 0.5 took to long to fit.
    # Other parameters are set according to the article recommended settings.
    n = X.shape[1]
    wolfs = (np.random.rand(n_agents, n) > .5).astype(float)  # initilize wolf pack

    fitnesses = calculate_fitnesses(X, y, wolfs, alpha)  # calculate initial fitness values
    sorted_index = np.argsort(fitnesses)
    x_alpha, x_beta, x_delta = [wolfs[i] for i in sorted_index[:3]]  # set alpha beta and delta wolfs

    min_fitness = 1
    min_fitness_x_alpha = -1

    for t in tqdm(range(iterations)):
        # calculate new wolf pack positions according to GWO algorithm:
        x_abd = np.stack([x_alpha, x_beta, x_delta]).copy()

        a = 2 - t * 2 / iterations
        A = np.abs(2 * a * np.random.rand(3, n) - a)
        C = 2 * np.random.rand(3, n)

        for wolf_ind in sorted_index:
            wolf = wolfs[wolf_ind]
            D = np.abs(C * x_abd - wolf)
            X_123 = x_abd - A * D
            wolfs[wolf_ind] = X_123.mean(axis=0)

        # apply sigmoid function over new position to convert them to binary
        x_si = 1 / (1 + np.exp(-wolfs))
        x_binary = np.random.rand(*wolfs.shape) >= x_si
        wolfs[:] = x_binary

        # calculate new positions fitnesses
        fitnesses = calculate_fitnesses(X, y, wolfs, alpha, two_phase_mutation_prob=two_phase_mutation_prob)
        sorted_index = np.argsort(fitnesses)
        x_alpha, x_beta, x_delta = [wolfs[i] for i in sorted_index[:3]] # update alpha beta and delta wolfs

        # save alpha if its the best seen
        if min(fitnesses) < min_fitness:
            min_fitness = min(fitnesses)
            min_fitness_x_alpha = x_alpha.copy()

    return min_fitness_x_alpha


def grey_wolf_fs_New(X, y, n_agents=5, iterations=30, alpha=0.01, two_phase_mutation_prob=0.1, n_layers=5):
    # implementation of TMGWO score function with our modifications.
    # The modification is that instead of alpha,beta,delta tagged wolfs we use "n_layers" tagged wolfs.
    # In our experiments we chose n_layers=n_agents so that all wolf are accounted for position updates.
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
        A = np.abs(2 * a * np.random.rand(n_layers, n) - a)
        C = 2 * np.random.rand(n_layers, n)

        for wolf_ind in sorted_index:
            wolf = wolfs[wolf_ind]
            D = np.abs(C * x_abd - wolf)
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
