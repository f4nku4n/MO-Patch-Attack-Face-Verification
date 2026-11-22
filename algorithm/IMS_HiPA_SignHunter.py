"""
    USE IMS + HiPA SignHunter 
    STEP 1: USE IMS_HIPA
    STEP 2: USE HiPA SignHunter 
    Step 3: USE HiPA 
"""

import math
from copy import deepcopy
from tqdm import tqdm

import torch

from core import Individual
from algorithm.HiPA import HiPA
from algorithm.HiPA_SignHunter_Test import SignHunterOptimizer

class SteppableSignHunter(SignHunterOptimizer):
    """SignHunterOptimizer.run() as a generator: each next() makes one query. No budget (the IMS loop controls it)."""

    def _eval_g(self, x):
        f = self._eval(x)
        yield
        return f

    def _refresh_cur_g(self):
        self.x_cur = self._make(self.sgn)
        if torch.equal(self.x_cur, self.x_base):
            self.f_cur = self.f_base
        else:
            self.f_cur = yield from self._eval_g(self.x_cur)

    def _hunt_g(self):
        for level in self.levels:
            order = torch.randperm(len(level)).tolist() if self.shuffle else range(len(level))
            for idx in order:
                g = level[idx]
                self._flip(g)
                x_new = self._make(self.sgn)
                if torch.equal(x_new, self.x_cur):
                    self._flip(g)
                    continue
                f_new = yield from self._eval_g(x_new)
                if f_new > self.f_cur:
                    self.x_cur, self.f_cur = x_new, f_new
                else:
                    self._flip(g)

    def _step_g(self):
        moved = False
        while self.f_cur > self.f_base:
            self.x_base, self.f_base = self.x_cur, self.f_cur
            self.n_steps += 1
            moved = True
            self.x_cur = self._make(self.sgn)
            if torch.equal(self.x_cur, self.x_base):
                self.f_cur = self.f_base
                break
            self.f_cur = yield from self._eval_g(self.x_cur)
            if not self.reuse_sign:
                break
        return moved

    def run_steps(self):
        self.f_base = yield from self._eval_g(self.x_base)
        yield from self._refresh_cur_g()
        while True:
            yield from self._hunt_g()
            self.n_passes += 1
            moved = yield from self._step_g()
            if not moved and self.eps_decay is not None and self.eps > self.eps_min:
                self.eps = max(self.eps * self.eps_decay, self.eps_min)
                yield from self._refresh_cur_g()


class SignHunterAttacker:
    """One IMS attacker = one patch size: its best individual + a paused SignHunter."""

    def __init__(self, algo, idv, x0, sh_kwargs):
        self.algo = algo
        self.best_idv = idv
        self.patch_size = idv.patch_size
        self.opt = SteppableSignHunter(x0, self._query, budget=math.inf, **sh_kwargs)
        self.gen = self.opt.run_steps()

    @property
    def score(self):
        return self.best_idv.adv_score.item()

    def _query(self, patch):
        idv = deepcopy(self.best_idv)
        idv.patch = patch.clone()
        self.algo.evaluate(idv)
        if idv.adv_score > self.best_idv.adv_score:
            self.best_idv = idv
        return idv.adv_score.item()

    def step(self):
        next(self.gen)


class IMS_HiPA_SignHunter(HiPA):
    def __init__(self, max_query, img_h, img_w, fitness, b=2, patch_s0=8, patch_ds=2, max_patch_s=None,
                 variant1=False, n_warmup=1, early_stop=False, eps=32 / 255, init='clean', min_block=2,
                 split_channels=True, reuse_sign=True, shuffle=True, eps_decay=0.5, eps_min=1 / 255):
        super().__init__(max_query=max_query, img_h=img_h, img_w=img_w, patch_s=patch_s0, fitness=fitness,
                         n_warmup=n_warmup, early_stop=early_stop)
        self.b = b
        self.patch_s0, self.patch_ds = patch_s0, patch_ds
        self.max_patch_s = max_patch_s if max_patch_s is not None else min(img_h, img_w)
        self.variant1 = variant1
        self.init = init
        self.sh_kwargs = dict(eps=eps, min_block=min_block, split_channels=split_channels, reuse_sign=reuse_sign,
                              shuffle=shuffle, eps_decay=eps_decay, eps_min=eps_min)
        self.budget = max_query - 20  # keep 20 queries for Step-3 (same as HiPA)

        self.attackers = []
        self.found, self.min_query = False, max_query
        self.started_sizes = []  # patch sizes that were initialised
        self.killed_sizes = []   # patch sizes killed by a larger patch


    ################################################### Query helper ###################################################
    def evaluate(self, idv):
        self.fitness.evaluate([idv])
        self._log([idv])
        self.prev_n_eval = self.fitness.n_eval
        if idv.adv_score >= 0 and not self.found:
            self.found, self.min_query = True, self.fitness.n_eval
######################################## Step-1: Promising Region Selection ########################################
    def _get_locations(self, patch_s):
        rows = sorted({min(i, self.img_h - patch_s) for i in range(0, self.img_h, patch_s)})
        cols = sorted({min(j, self.img_w - patch_s) for j in range(0, self.img_w, patch_s)})
        return [(i, i + patch_s, j, j + patch_s) for i in rows for j in cols]

    def _step1_cost(self, patch_s):
        return len(self._get_locations(patch_s)) * self.n_warmup

    def _promising_region_selection(self, patch_s):
        best_idv = None
        for loc in self._get_locations(patch_s):
            for _ in range(self.n_warmup):
                idv = Individual(patch_s, (self.img_h, self.img_w))
                idv.patch = self.get_init_patch(patch_s)
                idv.location = loc
                self.evaluate(idv)
                if best_idv is None or idv.adv_score > best_idv.adv_score:
                    best_idv = idv
        return best_idv

    def _init_content(self, idv):
        loc = idv.location
        if self.init == 'clean':
            return self.fitness.img1[:, loc[0]:loc[1], loc[2]:loc[3]].clone()
        if self.init == 'gray':
            return torch.full_like(idv.patch, 0.5)
        return idv.patch.clone()  # 'step1'

    def _add_attacker(self, patch_s):
        """Step-1 for a new patch size. Returns False if it does not fit (size or budget)."""
        if patch_s > self.max_patch_s or self._step1_cost(patch_s) > self.budget - self.fitness.n_eval:
            return False
        idv = self._promising_region_selection(patch_s)
        self.attackers.append(SignHunterAttacker(self, idv, self._init_content(idv), self.sh_kwargs))
        self.started_sizes.append(patch_s)
        return True

    ############################################ Step-2: Interleaved SignHunter ########################################
    def _interleave(self, n_step):
        i = 0
        while i < len(self.attackers) and self.fitness.n_eval < self.budget:
            if n_step % (self.b ** i) == 0:
                self.attackers[i].step()
                while i > 0 and self.attackers[i].score > self.attackers[i - 1].score:
                    self.killed_sizes.append(self.attackers[i - 1].patch_size)
                    del self.attackers[i - 1]
                    i -= 1
            i += 1

    ####################################################### Main #######################################################
    def solve(self):
        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        # Step 1 for the first patch size
        patch_s = self.patch_s0
        self._add_attacker(patch_s)

        # Step 2: IMS over SignHunter attackers (+ Step 1 of each new patch size)
        n_step, t, can_add = 1, 1, True
        while self.fitness.n_eval < self.budget and not (self.early_stop and self.found):
            n_step += 1
            self._interleave(n_step)
            if can_add and (self.variant1 or not self.found) and n_step == self.b ** t:
                patch_s += self.patch_ds
                can_add = self._add_attacker(patch_s)
                t += 1

        # Step 3: Stealth Refinement on the smallest surviving patch
        best_idv = self.attackers[0].best_idv
        self.patch_before_refining = deepcopy(best_idv)
        if not self.early_stop:
            best_idv = self._refine(best_idv)
        return best_idv, best_idv.patch_size, self.min_query

    def summary(self):
        return {
            'started_sizes': self.started_sizes,
            'killed_sizes': self.killed_sizes,
            'alive': [(a.patch_size, a.score, a.opt.n_passes, a.opt.n_steps, a.opt.eps) for a in self.attackers],
        }
