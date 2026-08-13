import torch
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from individual import Individual
from utils.evolutionary_algorithms import isBetter


class HillClimbing:
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, n_warmup=1, early_stop=False):
        self.max_query = max_query
        self.n_warmup = n_warmup
        self.img_h, self.img_w = img_h, img_w
        self.patch_s = patch_s

        self.fitness = fitness
        self.history = []

        self.early_stop = early_stop

    def _get_all_locations(self):
        list_locs = []
        for i in range(0, self.img_h, self.patch_s):
            for j in range(0, self.img_w, self.patch_s):
                list_locs.append((i, i + self.patch_s, j, j + self.patch_s))
        return list_locs

    @staticmethod
    def _add_rectangle(patch, patch_size):
        """
        Add a rectangle to the patch.
        """
        _patch = patch.clone()
        x_min = random.randint(0, patch_size - 1)
        y_min = random.randint(0, patch_size - 1)
        width = random.randint(2, 5)
        color = torch.rand(3).cuda()  # Random RGB color

        _patch[:, x_min: x_min + width, y_min: y_min + width] = color.unsqueeze(1).unsqueeze(2)
        return _patch

    def update_history(self, X):
        for idv in X:
            if idv.adv_score is not None:
                self.history.append([idv.adv_score.item(), idv.psnr_score.item()])

    def _log(self, X, pbar, prev_n_eval):
        self.update_history(X)
        pbar.update(self.fitness.n_eval - prev_n_eval)
        pbar.set_postfix(query=self.fitness.n_eval)

    def solve(self):
        pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        prev_n_eval = self.fitness.n_eval

        list_locs = self._get_all_locations()
        results = {loc: {} for loc in list_locs}
        loc2idx = {loc: i for i, loc in enumerate(list_locs)}

        X = []
        for loc in list_locs:
            _X = [Individual(self.patch_s, (self.img_h, self.img_w)) for _ in range(self.n_warmup)]
            for idv in _X:
                idv.location = loc
            self.fitness.evaluate(_X)
            self._log(_X, pbar, prev_n_eval)
            prev_n_eval = self.fitness.n_eval

            idx_best = 0
            for j in range(1, len(_X)):
                if isBetter(_X[idx_best], _X[j]):
                    idx_best = j
            list_adv_scores = [idv.adv_score.item() for idv in _X]
            results[loc] = np.mean(list_adv_scores)
            X.append(_X[idx_best])

        score_matrix = np.zeros((self.img_h // self.patch_s, self.img_h // self.patch_s))
        for loc in list_locs:
            i, j = loc[0] // self.patch_s, loc[2] // self.patch_s
            score_matrix[i][j] = results[loc]

        best_id = np.argwhere(score_matrix == np.max(score_matrix))[-1]
        min_x = int(best_id[0] * self.patch_s)
        max_x = min_x + self.patch_s
        min_y = int(best_id[1] * self.patch_s)
        max_y = min_y + self.patch_s
        loc_best_id = (min_x, max_x, min_y, max_y)
        # print(best_id, loc_best_id, np.max(score_matrix))

        best_idv = X[loc2idx[loc_best_id]]
        best_patch = best_idv.patch

        adv_score = np.max(score_matrix)
        # print('CurrentScore:', adv_score, best_idv.adv_score)

        while self.fitness.n_eval < self.max_query:
            new_idv = deepcopy(best_idv)
            new_patch = self._add_rectangle(best_patch, self.patch_s)
            new_idv.patch = new_patch

            self.fitness.evaluate([new_idv])
            self._log([new_idv], pbar, prev_n_eval)
            prev_n_eval = self.fitness.n_eval

            if isBetter(best_idv, new_idv):
                best_idv = new_idv
                best_patch = new_patch

            if self.early_stop and best_idv.adv_score >= 0:
                break
        return best_idv
