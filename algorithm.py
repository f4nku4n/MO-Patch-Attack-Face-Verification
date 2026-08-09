import torch
import random
import numpy as np
from copy import deepcopy

from utils.evolutionary_algorithms import tournament_selection, RankAndCrowdingSurvival

class GA:
    def __init__(self, max_iter, max_query, population, fitness, tournament_size=2, crossover_type='Blended',
                 terminated_condition='generation', problem_type='maximizing', using_rules=False):
        self.max_iter = max_iter
        self.max_query = max_query
        self.pop = population
        self.tournament_size = tournament_size

        self.fitness = fitness

        self.crossover_type = crossover_type
        self.history = []
        self.n_gen = 0

        self.using_rules = using_rules
        self.problem_type = problem_type
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
                _X.sort(key=lambda x: x.F, reverse=(self.problem_type == 'maximizing'))  # reverse=True for descending
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

class NSGAII(GA):
    def __init__(self, max_iter, max_query, population, fitness, crossover_type,
                 terminated_condition='generation', problem_type='minimizing'):
        super().__init__(max_iter, max_query, population, fitness, -1, crossover_type, terminated_condition, problem_type)
        self.selector = RankAndCrowdingSurvival()

    def selection(self, pool, n_survival, **kwargs):
        self.fitness.evaluate(pool)
        X_new = self.selector.do(pool, self.pop.pop_size, problem_type=self.problem_type)
        return X_new
