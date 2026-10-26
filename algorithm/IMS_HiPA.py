import torch
import random
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy

from core import Individual
from utils.evolutionary_algorithms import isBetter


class IMS_HiPA:
    def __init__(self, max_query, img_h, img_w, fitness, b, variant1=False, variant2=False, n_warmup=1,
                 early_stop=False, step2='rect', p_init=0.8):
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

        # Step-2 patch content: 'rect' = original HiPA (_add_rectangle),
        # 'patchrs' = PatchRS-style square updates (as in HiPA.py on branch bb_baselines)
        assert step2 in ('rect', 'patchrs'), step2
        self.step2 = step2
        self.p_init = p_init
        self.hc_it = {}         # patch_s -> number of hill-climbing steps done by that HiPA instance
        self.it_start_cu = {}   # patch_s -> step at which single-channel updates start

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
        width = random.randint(math.ceil(patch_size * 0.18), math.floor(patch_size * 0.36))
        color = torch.rand(3).cuda()  # Random RGB color

        _patch[:, x_min: x_min + width, y_min: y_min + width] = color.unsqueeze(1).unsqueeze(2)
        return _patch

    @staticmethod
    def _success(idv):
        succ = getattr(idv, 'success', None)
        return bool(succ) if succ is not None else bool(idv.adv_score >= 0)

    @staticmethod
    def _key(idv):
        """Lexicographic key (n_success, deficit, margin), or None when there is no such key
        (single-reference fitness, or a multi-reference run with --agg != lexi)."""
        return getattr(idv, 'key', None)

    @classmethod
    def _better(cls, a, b):
        """Is patch a better than patch b?
        With the lexicographic key: compare the number of fooled references first, then the
        remaining deficit, and only then the margin (Python compares tuples in that order).
        Otherwise fall back to the scalar adv_score, as in the original HiPA."""
        ka, kb = cls._key(a), cls._key(b)
        if ka is None or kb is None:
            return a.adv_score > b.adv_score
        return ka > kb
    
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

    ############################################ PatchRS-style patch content ###########################################
    def p_selection(self, it):
        """ schedule to decrease the parameter p (PatchRS / Square Attack) """
        it = int(it / self.max_query * 10000)
        p_init = self.p_init
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

    def _square_size(self, it, patch_s):
        return int(max(self.p_selection(it) ** .5 * patch_s, 1))

    def _get_it_start_cu(self, patch_s):
        if patch_s not in self.it_start_cu:
            it = 0
            for it in range(self.max_query):
                if self._square_size(it, patch_s) == 1:
                    break
            self.it_start_cu[patch_s] = it + (self.max_query - it) // 2
        return self.it_start_cu[patch_s]

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
            patch_univ[:, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = \
                self.random_choice([3, 1, 1])
        return patch_univ.clamp(0., 1.)

    def _patchrs_update(self, patch, patch_s):
        """One PatchRS-style perturbation of the patch content. Each HiPA instance (patch size) keeps its own
        step counter, so a HiPA created late in IMS still starts with large squares."""
        it = self.hc_it.get(patch_s, 0)
        self.hc_it[patch_s] = it + 1
        it_start_cu = self._get_it_start_cu(patch_s)

        s_it = self._square_size(it, patch_s)
        p_it = torch.randint(patch_s - s_it + 1, size=[2])
        new_patch = patch.clone()
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
            # single channel updates (s_it is 1 here)
            new_ch = self.random_int(low=0, high=3, shape=[1])
            new_patch[new_ch, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] = \
                1. - new_patch[new_ch, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it]
        return new_patch.clamp_(0., 1.)

    ######################################## Step-1: Promising Region Selection ########################################
    def _promising_region_selection(self, patch_s):
        list_locs = self._get_all_locations(patch_s)
        results = {loc: {} for loc in list_locs}
        loc2idx = {loc: i for i, loc in enumerate(list_locs)}

        X = []
        for loc in list_locs:
            _X = [Individual(patch_s, (self.img_h, self.img_w)) for _ in range(self.n_warmup)]
            for idv in _X:
                if self.step2 == 'patchrs':
                    idv.patch = self.get_init_patch(patch_s)
                idv.location = loc
            self.fitness.evaluate(_X)
            self._log()
            self.prev_n_eval = self.fitness.n_eval

            if self._key(_X[0]) is not None:        # lexicographic: score of the cell = key of its best patch
                idx_best = 0
                for j in range(1, len(_X)):
                    if self._better(_X[j], _X[idx_best]):
                        idx_best = j
                results[loc] = self._key(_X[idx_best])
            else:                                   # original behaviour (single reference / other --agg)
                idx_best = 0
                for j in range(1, len(_X)):
                    if isBetter(_X[idx_best], _X[j]):
                        idx_best = j
                results[loc] = np.mean([idv.adv_score.item() for idv in _X])
            X.append(_X[idx_best])

        loc_best_id = max(results, key=results.get)

        best_idv = X[loc2idx[loc_best_id]]
        best_patch = best_idv.patch

        return best_idv, best_patch

    ######################################## Step-2: Patch Content Optimization ########################################
    def _hillClimbing(self, best_idv, best_patch, patch_s):
        new_idv = deepcopy(best_idv)
        if self.step2 == 'patchrs':
            new_patch = self._patchrs_update(best_patch, patch_s)
        else:
            new_patch = self._add_rectangle(best_patch, patch_s)
        new_idv.patch = new_patch

        self.fitness.evaluate([new_idv])
        self._log()
        self.prev_n_eval = self.fitness.n_eval

        if self._better(new_idv, best_idv):  # Focus on finding a strong adversarial patch
            best_idv = new_idv
            best_patch = new_patch

        return best_idv, best_patch

    ############################################ Step-3: Stealth Refinement ############################################
    def _accept_refined(self, new_idv, best_idv, base_n):
        if base_n is not None:
            n = int(getattr(new_idv, 'n_success', 0))
            if n > base_n:
                return True
            if n < base_n:
                return False
        return bool(self._success(new_idv) and new_idv.psnr_score > best_idv.psnr_score)

    def _refine(self, idv):
        if self.fitness.n_eval >= self.max_query:
            return idv
        base_n = getattr(idv, 'n_success', None)   # None -> single-reference run, original behaviour
        if base_n is not None:
            base_n = int(base_n)
        new_w = 0.1
        best_idv = idv
        while new_w < self.w:
            new_patch = self._adjust_patch_with_weight(self.fitness.img1, idv.patch, idv.location, new_w)
            new_idv = deepcopy(idv)
            new_idv.patch = new_patch

            self.fitness.evaluate([new_idv])
            self._log()
            self.prev_n_eval = self.fitness.n_eval

            if self._accept_refined(new_idv, best_idv, base_n):
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

            if self._accept_refined(new_idv, best_idv, base_n):
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
        list_best_idv, list_best_patch = [], []

        cannot_add_new_patch = False
        # Init the first HiPA
        idv, patch = self._promising_region_selection(patch_s)
        if self._success(idv):
            found = True
            min_query = self.fitness.n_eval
        list_best_idv.append(idv)
        list_best_patch.append(patch)
        self.update_history(idv, patch_s)

        n_step = 1
        while (self.variant1 or not found) and self.fitness.n_eval < max_query:
            n_step += 1
            i = 0  # Index of the attacker in list_attackers

            while i < len(list_best_idv):
                if n_step % (self.b ** i) == 0:
                    patch_size = list_best_idv[i].patch_size
                    list_best_idv[i], list_best_patch[i] = self._hillClimbing(list_best_idv[i], list_best_patch[i],
                                                                              patch_size)
                    self.update_history(list_best_idv[i], patch_size)

                    while i > 0 and self._better(list_best_idv[i], list_best_idv[i - 1]):
                        # print(f'Kill HiPA (p={list_best_idv[i - 1].patch_size})')
                        del list_best_idv[i - 1]
                        del list_best_patch[i - 1]
                        i -= 1

                    if self._success(list_best_idv[i]):
                        if not found:
                            min_query = self.fitness.n_eval
                        found = True
                        if not self.variant1:
                            break
                i += 1
                if self.fitness.n_eval >= max_query:
                    break

            if (self.variant1 or not found) and n_step == self.b ** t:
                patch_s += 2
                query = math.ceil(self.img_h / patch_s) * math.ceil(self.img_w / patch_s) * self.n_warmup
                if query > max_query - self.fitness.n_eval:
                    # print(f'Cannot initialize HiPA (p={patch_s})')
                    cannot_add_new_patch = True
                    break
                else:
                    # print(f'Step {n_step}: Initialize HiPA (p={patch_s})')
                    idv, patch = self._promising_region_selection(patch_s)
                    self.update_history(idv, patch_s)
                    list_best_idv.append(idv)
                    list_best_patch.append(patch)
                    if self._success(idv):
                        if not found:
                            min_query = self.fitness.n_eval
                        found = True
                    t += 1
        # main
        while (not self.variant2 or cannot_add_new_patch) and self.fitness.n_eval < max_query:
            n_step += 1
            i = 0
            while i < len(list_best_idv):
                if n_step % (self.b ** i) == 0:
                    patch_size = list_best_idv[i].patch_size
                    list_best_idv[i], list_best_patch[i] = self._hillClimbing(list_best_idv[i], list_best_patch[i],
                                                                              patch_size)
                    self.update_history(list_best_idv[i], patch_size)
                    while i > 0 and self._better(list_best_idv[i], list_best_idv[i - 1]):
                        del list_best_idv[i - 1]
                        del list_best_patch[i - 1]
                        i -= 1
                    if self._success(list_best_idv[i]):
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
        patch_size = list_best_idv[0].patch_size
        while self.variant2 and found and self.fitness.n_eval < max_query:
            list_best_idv[0], list_best_patch[0] = self._hillClimbing(list_best_idv[0], list_best_patch[0], patch_size)
            self.update_history(list_best_idv[0], patch_size)

        # step 3
        patch = list_best_idv[0]
        self.patch_before_refining = deepcopy(patch)
        patch = self._refine(patch)
        return patch, patch_size, min_query