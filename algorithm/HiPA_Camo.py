import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from core import Individual
from utils.evolutionary_algorithms import isBetter
from algorithm.CamoPatch import render, mutate


class HiPA_Camo:
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, n_warmup=1, early_stop=False):
        self.max_query = max_query
        self.n_warmup = n_warmup
        self.img_h, self.img_w = img_h, img_w
        self.patch_s = patch_s

        self.fitness = fitness
        self.history = []

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

        s = self.patch_s

        X = []
        for loc in list_locs:
            _X = [Individual(self.patch_s, (self.img_h, self.img_w)) for _ in range(self.n_warmup)]
            for idv in _X:
                patch_geno = np.random.rand(100, 7)
                idv.set('patch_geno', patch_geno.copy())
                patch = torch.from_numpy(render(patch_geno, s, s).copy()).permute(2, 0, 1).cuda()
                idv.patch = patch.clone()
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

        best_idv = X[loc2idx[loc_best_id]]
        best_patch = best_idv.patch

        return best_idv, best_patch

    ######################################## Step-2: Patch Content Optimization ########################################
    def _hillClimbing(self, best_idv, best_patch, min_query, found):
        max_query = self.max_query
        best_patch_geno = best_idv.get('patch_geno').copy()
        s = self.patch_s
        while self.fitness.n_eval < max_query:
            new_patch_geno = mutate(best_patch_geno.copy(), 0.3)
            new_patch = torch.from_numpy(render(new_patch_geno, s, s).copy()).permute(2, 0, 1).cuda()

            new_idv = deepcopy(best_idv)
            new_idv.set('patch_geno', new_patch_geno)
            new_idv.patch = new_patch.clone()

            self.fitness.evaluate([new_idv])
            self._log([new_idv])
            self.prev_n_eval = self.fitness.n_eval

            if new_idv.adv_score >= 0 and not found:
                found = True
                max_query = self.max_query - 20
                min_query = self.fitness.n_eval

            if new_idv.adv_score > best_idv.adv_score:
                best_idv, best_patch_geno = new_idv, new_patch_geno

            if new_idv.adv_score >= 0 and self.fitness.n_eval >= self.max_query - 20:
                break

            if self.early_stop and best_idv.adv_score >= 0:
                break
        return best_idv, min_query, found

    ############################################ Step-3: Stealth Refinement ############################################
    def _refine(self, idv):
        if self.fitness.n_eval >= self.max_query:
            return idv
        new_w = 0.1
        best_idv = idv
        while new_w < self.w and self.fitness.n_eval < self.max_query:
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
        while new_w < self.w and self.fitness.n_eval < self.max_query:
            new_patch = self._adjust_patch_with_weight(self.fitness.img1, idv.patch, idv.location, new_w)
            new_idv = deepcopy(idv)
            new_idv.patch = new_patch

            self.fitness.evaluate([new_idv])
            self._log([new_idv])
            self.prev_n_eval = self.fitness.n_eval

            if new_idv.adv_score >= 0 and new_idv.psnr_score > best_idv.psnr_score:
                best_idv = new_idv
                break
            new_w = round(new_w + 0.01, 2)
        if self.fitness.n_eval < self.max_query:
            for _ in range(self.max_query - self.fitness.n_eval):
                self._log([best_idv])
        self.w = new_w
        return best_idv

    ####################################################### Main #######################################################
    def solve(self):
        found = False
        min_query = self.max_query

        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        # Step 1: Promising Region Selection
        best_idv, best_patch = self._promising_region_selection()
        if best_idv.adv_score >= 0 and not found:
            found = True
            min_query = self.fitness.n_eval

        # Step 2: Hill Climbing
        best_idv, min_query, found = self._hillClimbing(best_idv, best_patch, min_query, found)

        # Step 3: Stealth Refinement
        self.patch_before_refining = deepcopy(best_idv)
        best_idv = self._refine(best_idv)  # Enhance the stealth of found patch by blending it to the original content
        best_patch_size = best_idv.patch_size

        return best_idv, best_patch_size, min_query