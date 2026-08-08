import torch
import random
import numpy as np
from copy import deepcopy
from individual import Individual

from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from utils.evolutionary_algorithms import tournament_selection, calculating_crowding_distance

class GA:
    def __init__(self, max_iter, max_query, population, fitness, tournament_size=2, crossover_type='Blended',
                 terminated_condition='generation', using_rules=False):
        self.max_iter = max_iter
        self.max_query = max_query
        self.pop = population
        self.tournament_size = tournament_size

        self.fitness = fitness

        self.crossover_type = crossover_type
        self.history = []
        self.n_gen = 0

        self.using_rules = using_rules
        self.terminated_condition = terminated_condition

    def isTerminated(self):
        if self.terminated_condition == 'generation':
            return self.n_gen == self.max_iter
        elif self.terminated_condition == 'query':
            # TODO: Add the terminated condition that follows the maximum number of queries, self.max_query
            raise NotImplementedError("Chua cai dat")
        else:
            raise ValueError

    def update_history(self, X):
        for idv in X:
            if idv.adv_score is not None:
                self.history.append([idv.adv_score.item(), idv.psnr_score.item()])

    def solve(self):
        # TODO: Evaluate the fitness of initial population here (Using the self.fitness)
        # ...
        self.update_history(self.pop.P)

        self.n_gen = 1
        while not self.isTerminated():
            P = deepcopy(self.pop.P)
            O = []
            # Recombination
            for j in range(self.pop.pop_size // 2):
                ## Parent Selection
                ## Comment: Mama and Papa can be the same for two consecutive sampling. :)
                parent1, parent2 = random.sample(P, 2)

                ## Crossover
                if self.crossover_type == 'Blended':              
                    offspring_1, offspring_2 = parent1.crossover_blended(parent2)
                elif self.crossover_type == 'UX':
                    offspring_1, offspring_2 = parent1.crossover_UX(parent2)
                else:
                    raise ValueError

                ## Mutation
                offspring_1.mutate()
                offspring_2.mutate()
                
                O.append(offspring_1)
                O.append(offspring_2)
            PO = P + O

            self.pop.P = self.selection(pool=PO, n_survival=self.pop.pop_size, k=self.tournament_size)
            # TODO: Evaluate the fitness of offspring here (Using the self.fitness)
            # ...
            # self.pop.P = tournament_selection(pool=PO, k=self.tournament_size, n_survival=self.pop.pop_size)

            self.update_history(O)
            if self.has_converged(self.pop.P):
                print(f"Convergence reached at generation {self.n_gen}. Terminating early.")
                break

            self.n_gen += 1

        best_patch = self.return_best(self.pop.P)
        return best_patch
   
    @staticmethod
    def has_converged(population):
        """
        Check if all patches in the population are identical.
        If they are, the population has converged.
        """
        first_patch = population[0].patch
        for individual in population[1:]:
            if not torch.equal(first_patch, individual.patch):  # Check if patches are identical
                return False
        return True

    def selection(self, pool, n_survival, **kwargs):
        k = kwargs['k']
        if not self.using_rules:
            X_new = self.tournament_selection(pool, k, n_survival)
        else:
            X_new = self.tournament_selection_rules(pool, k, n_survival)
        return X_new

    def tournament_selection(self, pool, k, n_survival):
        self.fitness.evaluate(pool)
        X_new = []
        for _ in range(n_survival // (len(pool) // k)):
            random.shuffle(pool)
            for i in range(0, len(pool), k):
                _X = [pool[i + j] for j in range(k)]
                _X.sort(key=lambda x: x.F, reverse=True)  # reverse=True for descending
                X_new.append(_X[0])
        return X_new

    @staticmethod
    def isBetter(X, Y):
        adv_1, psnr_1 = X.adv_score, X.psnr_score
        adv_2, psnr_2 = Y.adv_score, Y.psnr_score
        if adv_1 >= 0 and adv_2 >= 0:
            return psnr_2 > psnr_1
        elif adv_1 < 0 and adv_2 > 0:
            return True
        elif adv_1 < 0 and adv_2 < 0:
            return adv_2 < adv_1
        return False

    def tournament_selection_rules(self, pool, k, n_survival):
        self.fitness.evaluate(pool)
        X_new = []
        for _ in range(n_survival // (len(pool) // k)):
            random.shuffle(pool)
            for i in range(0, len(pool), k):
                _X = [pool[i + j] for j in range(k)]
                best_idx = 0
                for j in range(1, len(_X)):
                    if self.isBetter(_X[best_idx], _X[j]):
                        best_idx = j
                X_new.append(_X[best_idx])
        return X_new

    @staticmethod
    def return_best(X):
        list_F = [idv.F.cpu() for idv in X]
        best_idx = np.argmax(list_F)
        best_patch = X[best_idx]
        return best_patch

selector = NonDominatedSorting()

class NSGAII(GA):
    def __init__(self, max_iter, max_query, population, fitness, crossover_type, terminated_condition='generation'):
        super().__init__(max_iter, max_query, population, fitness, crossover_type, terminated_condition)

    def update_archive(self, archive, new_ind: 'Individual'):

        if len(archive) == 0:
            return [new_ind]
        to_remove = []

        for i, item in enumerate(archive):
            dominant_status = self.is_dominant(item, new_ind)
            if dominant_status == 1:
                return archive

            elif dominant_status == 2:
                to_remove.append(i)
 
        for idx in reversed(to_remove):
            print("POP up")
            archive.pop(idx)
        archive.append(new_ind)

        return archive

    def selection(self, pool, n_survival, **kwargs):
        _, adv_scores, fsnr_scores = self.fitness.evaluate(pool)
        adv_scores_save = []
        psnr_scores_save = []

        # selection minimize for NSGAII
        F = np.array(torch.stack([-adv_scores, -fsnr_scores], dim=1).cpu().detach())
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=self.pop.pop_size)
        survivors = []

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calculating_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pool[i].rank = k
                pool[i].crowding = crowding_of_front[j]

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > self.pop.pop_size:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(self.pop.pop_size - len(survivors))]
 
            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
            index = list(front[I])
            adv_scores_save.append(adv_scores[index])
            psnr_scores_save.append(fsnr_scores[index])

            # adv_scores_save.append(pool[front[I]])
            # psnr_scores_save.append(pool[front[I]])
        
        self.archive.append({"adv_scores_log": adv_scores_save, "psnr_scores_log": psnr_scores_save})
        return [pool[i] for i in survivors]
