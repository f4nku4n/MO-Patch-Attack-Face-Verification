
import torch
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy

from functools import partial

from core import Individual
from utils.evolutionary_algorithms import isBetter


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


class SteppableSignHunter(SignHunterOptimizer):
    """SignHunterOptimizer.run() as a generator: each next() makes exactly one query (same as IMS_HiPA_SignHunter)."""

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


class HiPA_SignHunter_Test:
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, n_warmup=1, early_stop=False,
                 eps=32 / 255, init='clean', predefined_loc=None, min_block=2, split_channels=True,
                 reuse_sign=True, shuffle=True, eps_decay=0.5, eps_min=1 / 255):
        self.max_query = max_query
        self.n_warmup = n_warmup
        self.img_h, self.img_w = img_h, img_w
        self.patch_s = patch_s

        self.fitness = fitness
        self.history = {}

        self.early_stop = early_stop
        self.pbar = None
        self.prev_n_eval = 0

        self.patch_before_refining = None
        self.w = 1.0

        self.predefined_loc = predefined_loc  # (x_min, x_max, y_min, y_max) -> skip Step-1

        # SignHunter (Step-2)
        self.init = init  # 'clean' | 'step1' | 'gray'
        self.sh_kwargs = dict(eps=eps, min_block=min_block, split_channels=split_channels, reuse_sign=reuse_sign,
                              shuffle=shuffle, eps_decay=eps_decay, eps_min=eps_min)
        self.sh = {}  # patch size -> {'opt': SteppableSignHunter, 'gen': generator, 'idv': best individual}

        # Logging
        self.n_passes, self.n_steps, self.final_eps = 0, 0, eps

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

    def _fixed_region(self, patch_s):
        best_idv = Individual(patch_s, (self.img_h, self.img_w))
        best_idv.location = tuple(self.predefined_loc)
        self.fitness.evaluate([best_idv])
        self._log()
        self.prev_n_eval = self.fitness.n_eval

        best_patch = best_idv.patch
        return best_idv, best_patch

    ######################################## Step-2: Patch Content Optimization ########################################
    def _init_content(self, idv):
        loc = idv.location
        if self.init == 'clean':
            return self.fitness.img1[:, loc[0]:loc[1], loc[2]:loc[3]].clone()
        if self.init == 'gray':
            return torch.full_like(idv.patch, 0.5)
        return idv.patch.clone()  # 'step1'

    def _sh_query(self, patch_s, patch):
        state = self.sh[patch_s]
        new_idv = deepcopy(state['idv'])
        new_idv.patch = patch.clone()

        self.fitness.evaluate([new_idv])
        self._log()
        self.prev_n_eval = self.fitness.n_eval

        if new_idv.adv_score > state['idv'].adv_score:
            state['idv'] = new_idv
        return new_idv.adv_score.item()

    def _signHunter(self, best_idv, best_patch, patch_s):
        if patch_s not in self.sh:
            opt = SteppableSignHunter(self._init_content(best_idv), partial(self._sh_query, patch_s),
                                      budget=math.inf, **self.sh_kwargs)
            self.sh[patch_s] = {'opt': opt, 'gen': opt.run_steps(), 'idv': best_idv}
        next(self.sh[patch_s]['gen'])

        best_idv = self.sh[patch_s]['idv']
        return best_idv, best_idv.patch

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
        max_query = self.max_query - 20  # keep 20 queries for Step-3 (same as HiPA)
        patch_s = self.patch_s
        min_query = self.max_query
        found = False

        # Step 1
        if self.predefined_loc is not None:
            best_idv, best_patch = self._fixed_region(patch_s)
        else:
            best_idv, best_patch = self._promising_region_selection(patch_s)
        if best_idv.adv_score >= 0:
            found = True
            min_query = self.fitness.n_eval
        self.update_history(best_idv, patch_s)

        # Step 2: SignHunter
        while not (self.early_stop and found) and self.fitness.n_eval < max_query:
            best_idv, best_patch = self._signHunter(best_idv, best_patch, patch_s)
            self.update_history(best_idv, patch_s)
            if best_idv.adv_score >= 0 and not found:
                found = True
                min_query = self.fitness.n_eval

        if patch_s in self.sh:
            opt = self.sh[patch_s]['opt']
            self.n_passes, self.n_steps, self.final_eps = opt.n_passes, opt.n_steps, opt.eps

        # Step 3: Stealth Refinement
        self.patch_before_refining = deepcopy(best_idv)
        if not self.early_stop:
            best_idv = self._refine(best_idv)
        return best_idv, patch_s, min_query