import torch
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from core import Individual
from utils.evolutionary_algorithms import isBetter


class HillClimbing:
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, step1_random=False, step2_random=False, n_warmup=1, early_stop=False):
        self.max_query = max_query
        self.n_warmup = n_warmup
        self.img_h, self.img_w = img_h, img_w
        self.patch_s = patch_s

        self.fitness = fitness
        self.history = []

        self.step1_random, self.step2_random = step1_random, step2_random
        self.early_stop = early_stop
        self.pbar = None
        self.prev_n_eval = 0

        self.patch_before_refining = None
        self.w = 1.0

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

    def _log(self, X):
        self.update_history(X)
        self.pbar.update(self.fitness.n_eval - self.prev_n_eval)
        self.pbar.set_postfix(query=self.fitness.n_eval)

    @staticmethod
    def _adjust_patch_with_weight(cls_img, patch, loc, w):
        ori_patch = cls_img[:, loc[0]:loc[1], loc[2]:loc[3]].clone()
        new_patch = ori_patch * (1 - w) + patch * w
        return new_patch

    ######################################## Step-1: Promising Region Selection ########################################
    def _promising_region_selection(self):
        list_locs = self._get_all_locations()
        results = {loc: {} for loc in list_locs}
        loc2idx = {loc: i for i, loc in enumerate(list_locs)}

        X = []
        for loc in list_locs:
            _X = [Individual(self.patch_s, (self.img_h, self.img_w)) for _ in range(self.n_warmup)]
            for idv in _X:
                idv.location = loc
            self.fitness.evaluate(_X)
            self._log(_X)
            self.prev_n_eval = self.fitness.n_eval

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

        # adv_score = np.max(score_matrix)
        # print('CurrentScore:', adv_score, best_idv.adv_score)
        return best_idv, best_patch

    def _random_region(self):
        best_idv = Individual(self.patch_s, (self.img_h, self.img_w))
        best_patch = best_idv.patch

        self.fitness.evaluate([best_idv])
        self._log([best_idv])

        self.prev_n_eval = self.fitness.n_eval
        return best_idv, best_patch

    ######################################## Step-2: Patch Content Optimization ########################################
    def _hillClimbing(self, best_idv, best_patch):
        max_query = self.max_query
        found = False
        while self.fitness.n_eval < max_query:
            new_idv = deepcopy(best_idv)
            new_patch = self._add_rectangle(best_patch, self.patch_s)
            new_idv.patch = new_patch

            self.fitness.evaluate([new_idv])
            self._log([new_idv])
            self.prev_n_eval = self.fitness.n_eval

            if new_idv.adv_score >= 0 and not found:
                found = True
                max_query = self.max_query - 20

            if new_idv.adv_score > best_idv.adv_score:  # Focus on finding a fucking strong adversarial patch
                best_idv = new_idv
                best_patch = new_patch

            if self.early_stop and best_idv.adv_score >= 0:
                break
        return best_idv

    def _randomSearch(self, best_idv):
        n_candidates = self.max_query - self.fitness.n_eval - 20
        loc = best_idv.location
        list_candidates = [Individual(self.patch_s, (self.img_h, self.img_w)) for _ in range(n_candidates)]
        for idv in list_candidates:
            idv.location = loc
        i, j = 0, min(1000, len(list_candidates))
        while j != len(list_candidates):
            self.fitness.evaluate(list_candidates[i:j])
            i = j
            j = min(i + 1000, len(list_candidates))
        self.fitness.evaluate(list_candidates[i:j])
        self._log(list_candidates)
        self.prev_n_eval = self.fitness.n_eval

        for idv in list_candidates:
            if isBetter(best_idv, idv):
                best_idv = idv

        found = best_idv.adv_score >= 0
        if not found:
            n_candidates = 20
            list_candidates = [Individual(self.patch_s, (self.img_h, self.img_w)) for _ in range(n_candidates)]
            for idv in list_candidates:
                idv.location = loc
            self.fitness.evaluate(list_candidates)
            self._log(list_candidates)
            self.prev_n_eval = self.fitness.n_eval
            for idv in list_candidates:
                if isBetter(best_idv, idv):
                    best_idv = idv

        return best_idv

    ############################################ Step-3: Stealth Refinement ############################################
    def _refine(self, idv):
        if self.fitness.n_eval >= self.max_query:
            return idv
        new_w = 0.1
        best_idv = idv
        while new_w < self.w:
            new_patch = self._adjust_patch_with_weight(self.fitness.img1, idv.patch, idv.location, new_w)
            new_idv = deepcopy(idv)
            new_idv.patch = new_patch

            self.fitness.evaluate([new_idv])
            self._log([new_idv])
            self.prev_n_eval = self.fitness.n_eval

            if new_idv.adv_score >= 0 and new_idv.psnr_score > best_idv.psnr_score:
                best_idv = new_idv
                break
            new_w = round(new_w + 0.1, 1)

        self.w = new_w
        new_w = round(self.w - 0.1 + 0.01, 2)
        while new_w < self.w:
            new_patch = self._adjust_patch_with_weight(self.fitness.img1, idv.patch, idv.location, new_w)
            new_idv = deepcopy(idv)
            new_idv.patch = new_patch

            self.fitness.evaluate([new_idv])
            self._log([new_idv])
            self.prev_n_eval = self.fitness.n_eval

            if new_idv.adv_score >= 0 and new_idv.psnr_score > best_idv.psnr_score:
                best_idv = new_idv
                self.w = new_w
                break
            new_w = round(new_w + 0.01, 2)
        if self.fitness.n_eval < self.max_query:
            for _ in range(self.max_query - self.fitness.n_eval):
                self._log([best_idv])

        return best_idv

    ####################################################### Main #######################################################
    def solve(self):
        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        if self.step1_random:
            best_idv, best_patch = self._random_region()
        else:
            best_idv, best_patch = self._promising_region_selection()

        # Step 2: Hill Climbing
        if self.step2_random:
            best_idv = self._randomSearch(best_idv)
        else:
            best_idv = self._hillClimbing(best_idv, best_patch)

        # Step 3: Stealth Refinement
        self.patch_before_refining = deepcopy(best_idv)
        best_idv = self._refine(best_idv)  # Enhance the stealth of found patch by blending it to the original content
        return best_idv
