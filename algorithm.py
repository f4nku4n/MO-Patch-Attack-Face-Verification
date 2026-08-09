import torch
import random
import numpy as np
from copy import deepcopy

from utils.evolutionary_algorithms import tournament_selection, tournament_selection_rules, RankAndCrowdingSurvival

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
            return self.fitness.n_eval >= self.max_query
        else:
            raise ValueError

    def update_history(self, X):
        for idv in X:
            if idv.adv_score is not None:
                self.history.append([idv.adv_score.item(), idv.psnr_score.item()])

    def solve(self):
        self.fitness.evaluate(self.pop.P)
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

            self.fitness.evaluate(O)
            self.pop.P = self.selection(pool=PO, n_survival=self.pop.pop_size,
                                        k=self.tournament_size, problem_type=self.problem_type)

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
            X_new = tournament_selection(pool, k, n_survival, problem_type=kwargs['problem_type'])
        else:
            X_new = tournament_selection_rules(pool, k, n_survival)
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

    @staticmethod
    def return_best(pop):
        pass
        # list_F = [idv.F.cpu() for idv in X]
        # best_idx = np.argmax(list_F)
        # best_patch = X[best_idx]
        # return best_patch
