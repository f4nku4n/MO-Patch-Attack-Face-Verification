"""
HiPA + SignHunter (test version).

Step-1 (Promising Region Selection) and Step-3 (Stealth Refinement) are the same as HiPA.
Step-2 is SignHunter (Al-Dujaili & O'Reilly, ICLR 2020) with 4 changes w.r.t. HiPA_SignHunter.py:
    1. Sign reuse (line search): after a hunting pass, keep stepping  x_base <- clip(x_base + eps * sgn)
       (1 query / step) as long as the score improves. The sign is only re-estimated when the step fails.
       -> one hunting pass (~2n queries) gives many iFGSM-like steps instead of only one.
    2. Spatial groups: at level h the patch is split into a 2^h x 2^h grid of square blocks
       (all 3 channels of a block are flipped together) instead of chunks of the flattened vector.
    3. The hierarchy stops at blocks of side `min_block` (optionally one more level that splits the
       3 channels of each finest block), instead of going down to single coordinates.
    4. Step-size decay: if a whole pass + step brings no improvement, eps <- max(eps * eps_decay, eps_min).
The patch is NOT projected onto an eps-ball around the clean content (same as run_iFGSM).
"""
import math
import torch
from copy import deepcopy
from tqdm import tqdm

from core import Individual
from algorithm.HiPA import HiPA


class SignHunterOptimizer:
    """
    Framework-agnostic SignHunter (maximisation) so it can be unit-tested on CPU.
        x0        : tensor (C, H, W), starting point
        objective : callable(tensor) -> float, 1 call = 1 query
        budget    : max number of objective calls
        stop_fn   : callable(best_value) -> bool, early stop
    """
    def __init__(self, x0, objective, eps, budget, min_block=2, split_channels=True, reuse_sign=True,
                 shuffle=True, eps_decay=0.5, eps_min=1 / 255, stop_fn=None, lo=0., hi=1.):
        self.objective = objective
        self.eps, self.eps_decay, self.eps_min = eps, eps_decay, eps_min
        self.budget = budget
        self.min_block = max(1, min_block)
        self.split_channels = split_channels
        self.reuse_sign = reuse_sign
        self.shuffle = shuffle
        self.stop_fn = stop_fn
        self.lo, self.hi = lo, hi

        self.x_base = x0.clone()
        self.f_base = None
        self.sgn = torch.ones_like(x0)
        self.x_cur, self.f_cur = None, None      # current sign point: clip(x_base + eps * sgn)
        self.x_best, self.f_best = None, -math.inf

        self.n_q = 0
        self.n_passes = 0
        self.n_steps = 0
        self.levels = self._build_levels(x0.shape)

    ############################################### Groups ###############################################
    @staticmethod
    def _split(size, k):
        bounds = torch.linspace(0, size, k + 1).round().long().tolist()
        return [(bounds[i], bounds[i + 1]) for i in range(k) if bounds[i + 1] > bounds[i]]

    def _build_levels(self, shape):
        c, h, w = shape
        k_max = max(1, math.ceil(max(h, w) / self.min_block))
        levels, k = [], 1
        while True:
            k_used = min(k, k_max)
            rows, cols = self._split(h, k_used), self._split(w, k_used)
            levels.append([(0, c, r0, r1, c0, c1) for (r0, r1) in rows for (c0, c1) in cols])
            if k_used == k_max:
                break
            k *= 2
        if self.split_channels and c > 1:
            levels.append([(ch, ch + 1, r0, r1, c0, c1) for (_, _, r0, r1, c0, c1) in levels[-1] for ch in range(c)])
        return levels

    ############################################### Helpers ###############################################
    def _make(self, sgn):
        return (self.x_base + self.eps * sgn).clamp(self.lo, self.hi)

    def _done(self):
        return self.n_q >= self.budget or (self.stop_fn is not None and self.stop_fn(self.f_best))

    def _eval(self, x):
        f = self.objective(x)
        self.n_q += 1
        if f > self.f_best:
            self.x_best, self.f_best = x.clone(), f
        return f

    def _flip(self, g):
        ch0, ch1, r0, r1, c0, c1 = g
        self.sgn[ch0:ch1, r0:r1, c0:c1] *= -1

    def _refresh_cur(self):
        self.x_cur = self._make(self.sgn)
        if torch.equal(self.x_cur, self.x_base):
            self.f_cur = self.f_base
        elif not self._done():
            self.f_cur = self._eval(self.x_cur)
        else:
            self.f_cur = -math.inf

    ############################################### Phases ###############################################
    def _hunt(self):
        """One SignHunter pass: flip groups level by level, keep a flip only if the sign point improves."""
        for level in self.levels:
            order = torch.randperm(len(level)).tolist() if self.shuffle else range(len(level))
            for idx in order:
                if self._done():
                    return
                g = level[idx]
                self._flip(g)
                x_new = self._make(self.sgn)
                if torch.equal(x_new, self.x_cur):   # every flipped coordinate is clipped -> no change, save the query
                    self._flip(g)
                    continue
                f_new = self._eval(x_new)
                if f_new > self.f_cur:
                    self.x_cur, self.f_cur = x_new, f_new
                else:
                    self._flip(g)

    def _step(self):
        """Move the base point along the current sign while it improves. Returns True if it moved."""
        moved = False
        while self.f_cur > self.f_base:
            self.x_base, self.f_base = self.x_cur, self.f_cur
            self.n_steps += 1
            moved = True
            self.x_cur = self._make(self.sgn)
            if torch.equal(self.x_cur, self.x_base):
                self.f_cur = self.f_base
                break
            if self._done():
                self.f_cur = -math.inf
                break
            self.f_cur = self._eval(self.x_cur)
            if not self.reuse_sign:
                break
        return moved

    def run(self):
        self.f_base = self._eval(self.x_base)
        self._refresh_cur()
        while not self._done():
            self._hunt()
            self.n_passes += 1
            moved = self._step()
            if not moved and self.eps_decay is not None and self.eps > self.eps_min and not self._done():
                self.eps = max(self.eps * self.eps_decay, self.eps_min)
                self._refresh_cur()
        return self.x_best, self.f_best


class HiPA_SignHunter_Test(HiPA):
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, n_warmup=1, early_stop=False,
                 eps=32 / 255, init='clean', predefined_loc=None, min_block=2, split_channels=True,
                 reuse_sign=True, shuffle=True, eps_decay=0.5, eps_min=1 / 255):
        super().__init__(max_query=max_query, img_h=img_h, img_w=img_w, patch_s=patch_s, fitness=fitness,
                         n_warmup=n_warmup, early_stop=early_stop)
        self.eps = eps
        self.init = init                      # 'clean' | 'step1' | 'gray'
        self.predefined_loc = predefined_loc  # (x_min, x_max, y_min, y_max) -> skip Step-1
        self.min_block = min_block
        self.split_channels = split_channels
        self.reuse_sign = reuse_sign
        self.shuffle = shuffle
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        # State of Step-2
        self.best_idv = None
        self.found = False
        self.min_query = max_query

        # Logging
        self.n_passes, self.n_steps, self.final_eps = 0, 0, eps

    ######################################## Step-1: Promising Region Selection ########################################
    def _promising_region_selection(self):
        list_locs = [loc for loc in self._get_all_locations()
                     if loc[1] <= self.img_h and loc[3] <= self.img_w]  # drop partial cells at the border
        X = []
        for loc in list_locs:
            _X = [Individual(self.patch_s, (self.img_h, self.img_w)) for _ in range(self.n_warmup)]
            for idv in _X:
                idv.patch = self.get_init_patch(self.patch_s).clone()
                idv.location = loc
            self.fitness.evaluate(_X)
            self._log(_X)
            self.prev_n_eval = self.fitness.n_eval
            X.append(max(_X, key=lambda idv: idv.adv_score.item()))
        return max(X, key=lambda idv: idv.adv_score.item())

    def _fixed_region(self):
        best_idv = Individual(self.patch_s, (self.img_h, self.img_w))
        best_idv.location = tuple(self.predefined_loc)
        best_idv.patch = self.get_init_patch(self.patch_s)
        self.fitness.evaluate([best_idv])
        self._log([best_idv])
        self.prev_n_eval = self.fitness.n_eval
        return best_idv

    ############################################ Step-2: SignHunter ####################################################
    def _query(self, patch):
        idv = deepcopy(self.best_idv)
        idv.patch = patch.clone()
        self.fitness.evaluate([idv])
        self._log([idv])
        self.prev_n_eval = self.fitness.n_eval
        if idv.adv_score >= 0 and not self.found:
            self.found, self.min_query = True, self.fitness.n_eval
        if idv.adv_score > self.best_idv.adv_score:
            self.best_idv = idv
        return idv.adv_score.item()

    def _stop(self, f_best):
        return self.early_stop and f_best >= 0

    def _init_patch(self):
        loc = self.best_idv.location
        if self.init == 'clean':
            return self.fitness.img1[:, loc[0]:loc[1], loc[2]:loc[3]].clone()
        if self.init == 'gray':
            return torch.full_like(self.best_idv.patch, 0.5)
        return self.best_idv.patch.clone()  # 'step1'

    def _signHunter(self):
        budget = self.max_query - self.fitness.n_eval - 20  # keep 20 queries for Step-3 (same as HiPA)
        opt = SignHunterOptimizer(self._init_patch(), self._query, self.eps, budget,
                                  min_block=self.min_block, split_channels=self.split_channels,
                                  reuse_sign=self.reuse_sign, shuffle=self.shuffle,
                                  eps_decay=self.eps_decay, eps_min=self.eps_min, stop_fn=self._stop)
        opt.run()
        self.n_passes, self.n_steps, self.final_eps = opt.n_passes, opt.n_steps, opt.eps

    ####################################################### Main #######################################################
    def solve(self):
        self.found = False
        self.min_query = self.max_query

        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        # Step 1
        if self.predefined_loc is not None:
            self.best_idv = self._fixed_region()
        else:
            self.best_idv = self._promising_region_selection()
        if self.best_idv.adv_score >= 0:
            self.found, self.min_query = True, self.fitness.n_eval

        # Step 2: SignHunter
        if not (self.early_stop and self.found):
            self._signHunter()

        # Step 3: Stealth Refinement
        best_idv = self.best_idv
        self.patch_before_refining = deepcopy(best_idv)
        if not self.early_stop:
            best_idv = self._refine(best_idv)
        return best_idv, best_idv.patch_size, self.min_query