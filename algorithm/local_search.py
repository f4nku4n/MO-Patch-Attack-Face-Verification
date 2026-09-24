import torch
import random
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy

from core import Individual
from utils.evolutionary_algorithms import isBetter


class HillClimbing:
    def __init__(self, max_query, img_h, img_w, fitness, b, variant1=False, variant2=False, n_warmup=1,
                 early_stop=False):
        self.max_query = max_query
        self.n_warmup = n_warmup
        self.img_h, self.img_w = img_h, img_w

        self.fitness = fitness
        self.history = {}

        self.variant1, self.variant2 = variant1, variant2
        self.early_stop = early_stop
        self.pbar = None
        self.prev_n_eval = 0

        self.patch_before_refining = None
        self.w = 1.0
        self.b = b
    def _get_all_locations(self, patch_s):
        list_locs = []
        for i in range(0, self.img_h, patch_s):
            if i + patch_s > self.img_h:
                i = self.img_h - patch_s
            for j in range(0, self.img_w, patch_s):
                if j + patch_s > self.img_w:
                    j = self.img_w - patch_s
                list_locs.append((i, i + patch_s, j, j + patch_s))
        return list_locs

    @staticmethod
    def _add_rectangle(patch, patch_size):
        """
        Add a rectangle to the patch.
        """
        _patch = patch.clone()
        x_min = random.randint(0, patch_size - 1)
        y_min = random.randint(0, patch_size - 1)
        width = random.randint(math.ceil(patch_size*0.18), math.floor(patch_size*0.36))
        color = torch.rand(3).cuda()  # Random RGB color

        _patch[:, x_min: x_min + width, y_min: y_min + width] = color.unsqueeze(1).unsqueeze(2)
        return _patch

    def update_history(self, idv, patch_s):
        if patch_s not in self.history:
            self.history[patch_s] = [[idv.adv_score.item(), idv.psnr_score.item()]]
        else:
            self.history[patch_s].append([idv.adv_score.item(), idv.psnr_score.item()])

    def _log(self):
        self.pbar.update(self.fitness.n_eval - self.prev_n_eval)
        self.pbar.set_postfix(query=self.fitness.n_eval)

    @staticmethod
    def _adjust_patch_with_weight(cls_img, patch, loc, w):
        ori_patch = cls_img[:, loc[0]:loc[1], loc[2]:loc[3]].clone()
        new_patch = ori_patch * (1 - w) + patch * w
        return new_patch

    ######################################## Step-1: Promising Region Selection ########################################
    def _promising_region_selection(self, patch_s):
        list_locs = self._get_all_locations(patch_s)
        results = {loc: {} for loc in list_locs}
        loc2idx = {loc: i for i, loc in enumerate(list_locs)}

        X = []
        for loc in list_locs:
            _X = [Individual(patch_s, (self.img_h, self.img_w)) for _ in range(self.n_warmup)]
            for idv in _X:
                idv.location = loc
            self.fitness.evaluate(_X)
            self._log()
            self.prev_n_eval = self.fitness.n_eval

            idx_best = 0
            for j in range(1, len(_X)):
                if isBetter(_X[idx_best], _X[j]):
                    idx_best = j
            list_adv_scores = [idv.adv_score.item() for idv in _X]
            results[loc] = np.mean(list_adv_scores)
            X.append(_X[idx_best])

        loc_best_id = max(results, key=results.get)

        best_idv = X[loc2idx[loc_best_id]]
        best_patch = best_idv.patch

        return best_idv, best_patch


    ######################################## Step-2: Patch Content Optimization ########################################
    def _hillClimbing(self, best_idv, best_patch, patch_s):
        new_idv = deepcopy(best_idv)
        new_patch = self._add_rectangle(best_patch, patch_s)
        new_idv.patch = new_patch

        self.fitness.evaluate([new_idv])
        self._log()
        self.prev_n_eval = self.fitness.n_eval

        if new_idv.adv_score > best_idv.adv_score:  # Focus on finding a fucking strong adversarial patch
            best_idv = new_idv
            best_patch = new_patch

        return best_idv, best_patch


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
            self._log()
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
            self._log()
            self.prev_n_eval = self.fitness.n_eval

            if new_idv.adv_score >= 0 and new_idv.psnr_score > best_idv.psnr_score:
                best_idv = new_idv
                self.w = new_w
                break
            new_w = round(new_w + 0.01, 2)
        if self.fitness.n_eval < self.max_query:
            for _ in range(self.max_query - self.fitness.n_eval):
                self._log()

        return best_idv

    ####################################################### Main #######################################################
    def solve(self):
        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval
        max_query = self.max_query - 20
        t = 1
        patch_s = 8
        min_query = self.max_query
        found = False
        best_idv = []
        best_patch = []
        cannot_add_new_patch = False
        idv, patch = self._promising_region_selection(patch_s)
        if idv.adv_score >= 0:
            found = True
            min_query = self.fitness.n_eval
        best_idv.append(idv)
        best_patch.append(patch)
        self.update_history(idv, patch_s)
        n_step = 1
        # variant1
        while (self.variant1 or not found) and self.fitness.n_eval < max_query:
            n_step += 1
            i = 0
            while i < len(best_idv):
                if n_step % (self.b ** i) == 0:
                    patch_size = best_idv[i].patch_size
                    best_idv[i], best_patch[i] = self._hillClimbing(best_idv[i], best_patch[i], patch_size)
                    self.update_history(best_idv[i], patch_size)
                    while i > 0 and best_idv[i].adv_score > best_idv[i - 1].adv_score:
                        del best_idv[i - 1]
                        del best_patch[i - 1]
                        i -= 1

                    if best_idv[i].adv_score >= 0:
                        if not found:
                            min_query = self.fitness.n_eval
                        found = True
                        if not self.variant1:
                            break
                i += 1
                if self.fitness.n_eval >= max_query:
                    break
            if (self.variant1 or not found) and n_step == self.b**t:
                patch_s += 2
                query = math.ceil(self.img_h / patch_s) * math.ceil(self.img_w / patch_s) * self.n_warmup
                if query > max_query - self.fitness.n_eval:
                    cannot_add_new_patch = True
                    break
                else:
                    idv, patch = self._promising_region_selection(patch_s)
                    self.update_history(idv, patch_s)
                    best_idv.append(idv)
                    best_patch.append(patch)
                    if idv.adv_score >= 0:
                        if not found:
                            min_query = self.fitness.n_eval
                        found = True    
                    t += 1
        # main
        while (not self.variant2 or cannot_add_new_patch) and self.fitness.n_eval < max_query:
            n_step += 1
            i = 0
            while i < len(best_idv):
                if n_step % (self.b ** i) == 0:
                    patch_size = best_idv[i].patch_size
                    best_idv[i], best_patch[i] = self._hillClimbing(best_idv[i], best_patch[i], patch_size)
                    self.update_history(best_idv[i], patch_size)
                    while i > 0 and best_idv[i].adv_score > best_idv[i - 1].adv_score:
                        del best_idv[i - 1]
                        del best_patch[i - 1]
                        i -= 1
                    if best_idv[i].adv_score >= 0:
                        if not found:
                            min_query = self.fitness.n_eval
                        found = True
                        if self.variant2:
                            cannot_add_new_patch = False
                            break
                i += 1
                if self.fitness.n_eval >= max_query:
                    break

        # variant2
        patch_size = best_idv[0].patch_size
        while self.variant2 and found and self.fitness.n_eval < max_query:
            best_idv[0], best_patch[0] = self._hillClimbing(best_idv[0], best_patch[0], patch_size)
            self.update_history(best_idv[0], patch_size)

        # step 3
        patch = best_idv[0]
        self.patch_before_refining = deepcopy(patch)
        patch = self._refine(patch)
        return patch, patch_size, min_query

