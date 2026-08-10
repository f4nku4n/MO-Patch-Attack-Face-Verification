import random
import numpy as np

from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

## For Single-Objective Optimization
def tournament_selection(pool, k, n_survival, problem_type='maximizing'):
    X_new = []
    for _ in range(n_survival // (len(pool) // k)):
        random.shuffle(pool)
        for i in range(0, len(pool), k):
            _X = [pool[i + j] for j in range(k)]
            _X.sort(key=lambda x: x.F, reverse=(problem_type == 'maximizing'))  # reverse=True for maximizing problem
            X_new.append(_X[0])
    return X_new

def isBetter(X, Y):
    adv_1, psnr_1 = X.adv_score, X.psnr_score
    adv_2, psnr_2 = Y.adv_score, Y.psnr_score
    if adv_1 >= 0 and adv_2 >= 0:
        return psnr_2 > psnr_1
    elif adv_1 < 0 and adv_2 > 0:
        return True
    elif adv_1 < 0 and adv_2 < 0:
        return adv_2 > adv_1
    return False

def tournament_selection_rules(pool, k, n_survival):
    X_new = []
    for _ in range(n_survival // (len(pool) // k)):
        random.shuffle(pool)
        for i in range(0, len(pool), k):
            _X = [pool[i + j] for j in range(k)]
            best_idx = 0
            for j in range(1, len(_X)):
                if isBetter(_X[best_idx], _X[j]):
                    best_idx = j
            X_new.append(_X[best_idx])
    return X_new

## For Multi-Objective Optimization
class RankAndCrowdingSurvival:
    def __init__(self):
        self.name = 'Rank and Crowding Survival'

    @staticmethod
    def do(pop, n_survive, problem_type='minimizing'):
        pop = np.array(pop)

        # get the objective space values and objects
        F = np.array([idv.F for idv in pop])

        # Only supported for minimizing problem. If the problem is maximizing, multiply -1 to the fitness score
        if problem_type == 'maximizing':
            F = F * -1

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calculating_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set('rank', k)
                pop[i].set('crowding', crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
        return pop[survivors].tolist()


def calculating_crowding_distance(F):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity
    return crowding
