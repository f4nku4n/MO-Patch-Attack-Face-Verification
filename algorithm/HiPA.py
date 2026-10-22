import torch
import random
import math
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from core import Individual
from utils.evolutionary_algorithms import isBetter


class HiPA:
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

    def p_selection(self, it):
        """ schedule to decrease the parameter p """
        it = int(it / self.max_query * 10000)
        p_init = 0.8
        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it:
            p = p_init / 512
        else:
            p = p_init
        return p

    def sh_selection(self, it):
        """ schedule to decrease the parameter p """
        t = max((float(self.max_query - it) / self.max_query - .0) ** 1., 0) * .75
        return t

    @staticmethod
    def random_choice(shape):
        t = 2 * torch.rand(shape).to('cuda') - 1
        return torch.sign(t)

    @staticmethod
    def random_int(low=0, high=1, shape=(1, )):
        t = low + (high - low) * torch.rand(shape).to('cuda')
        return t.long()

    def get_init_patch(self, s, n_iter=1000):
        patch_univ = torch.zeros([3, s, s]).to('cuda')
        for _ in range(n_iter):
            size_init = torch.randint(low=1, high=math.ceil(s ** .5), size=[1]).item()
            loc_init = torch.randint(s - size_init + 1, size=[2])
            patch_univ[:, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = self.random_choice([3, 1, 1])
        return patch_univ.clamp(0., 1.)

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
                patch = self.get_init_patch(self.patch_s)
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
    def _hillClimbing(self, best_idv, best_patch, min_query, found):
        max_query = self.max_query

        it = 0
        for it in range(self.max_query):
            s_it = int(max(self.p_selection(it) ** .5 * self.patch_s, 1))
            if s_it == 1:
                break
        it_start_cu = it + (self.max_query - it) // 2

        while self.fitness.n_eval < max_query:
            s_it = int(max(self.p_selection(it) ** .5 * self.patch_s, 1))
            p_it = torch.randint(self.patch_s - s_it + 1, size=[2])

            new_patch = best_patch.clone()
            if it < it_start_cu:
                if s_it > 1:
                    new_patch[:, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] += self.random_choice([3, 1, 1])
                else:
                    old_clr = new_patch[:, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it].clone()
                    new_clr = old_clr.clone()
                    while (new_clr == old_clr).all().item():
                        new_clr = self.random_choice([3, 1, 1]).clone().clamp(0., 1.)
                    new_patch[:, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] = new_clr.clone()
            else:
                assert s_it == 1
                # single channel updates
                new_ch = self.random_int(low=0, high=3, shape=[1])
                new_patch[new_ch, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] = 1. - new_patch[
                                                                                         new_ch,
                                                                                         p_it[0]:p_it[0] + s_it,
                                                                                         p_it[1]:p_it[1] + s_it]
            new_patch.clamp_(0., 1.)

            new_idv = deepcopy(best_idv)
            new_idv.patch = new_patch

            self.fitness.evaluate([new_idv])
            self._log([new_idv])
            self.prev_n_eval = self.fitness.n_eval

            if new_idv.adv_score >= 0 and not found:
                found = True
                max_query = self.max_query - 20
                min_query = self.fitness.n_eval

            if new_idv.adv_score > best_idv.adv_score:
                best_idv, best_patch = new_idv, new_patch

            if new_idv.adv_score >= 0 and self.fitness.n_eval >= self.max_query - 20:
                break

            if self.early_stop and best_idv.adv_score >= 0:
                break
        return best_idv, min_query, found

    def _randomSearch(self, best_idv, min_query, found):
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

        if self.step1_random:
            best_idv, best_patch = self._random_region()
        else:
            best_idv, best_patch = self._promising_region_selection()
        if best_idv.adv_score >= 0 and not found:
            found = True
            min_query = self.fitness.n_eval

        # Step 2: Hill Climbing
        if self.step2_random:
            best_idv, min_query, found = self._randomSearch(best_idv, min_query, found)
        else:
            best_idv, min_query, found = self._hillClimbing(best_idv, best_patch, min_query, found)

        # Step 3: Stealth Refinement
        self.patch_before_refining = deepcopy(best_idv)
        best_idv = self._refine(best_idv)  # Enhance the stealth of found patch by blending it to the original content
        best_patch_size = best_idv.patch_size

        return best_idv, best_patch_size, min_query