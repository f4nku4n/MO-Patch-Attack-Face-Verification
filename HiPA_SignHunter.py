"""
HiPA + SignHunter (Al-Dujaili & O'Reilly, ICLR 2020, "Sign Bits Are All You Need for Black-Box Attacks").

Step-1 (Promising Region Selection) and Step-3 (Stealth Refinement) are inherited from HiPA.
Step-2 (Patch Content Optimization) is replaced by an *iterative* SignHunter:
    - SignHunter estimates sign(grad) of the adversarial score w.r.t. the patch pixels by flipping
      the sign of groups of coordinates (halving the group size at every level) and keeping a flip
      only if it increases the score. This mimics the update direction of the white-box iFGSM.
    - When all coordinates have been visited at the finest level, the current best patch becomes the
      new base point and the procedure starts again (like the next step of iFGSM).
The patch is NOT constrained to an eps-ball around the clean content (same as run_iFGSM: projection
is commented out there); `eps` acts as the step size alpha of iFGSM.
"""
import math
import torch
from copy import deepcopy

from core import Individual
from algorithm.HiPA import HiPA


class HiPA_SignHunter(HiPA):
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, n_warmup=1, early_stop=False,
                 eps=32 / 255, init='clean', predefined_loc=None):
        super().__init__(max_query=max_query, img_h=img_h, img_w=img_w, patch_s=patch_s, fitness=fitness,
                         n_warmup=n_warmup, early_stop=early_stop)
        self.eps = eps
        self.init = init                      # 'clean' (same as iFGSM) | 'step1' (random patch found by Step-1)
        self.predefined_loc = predefined_loc  # (x_min, x_max, y_min, y_max) -> skip Step-1 (e.g. reuse iFGSM loc)
        self.n_restarts = 0                   # number of finished SignHunter passes (for logging)

    ######################################## Step-1: Promising Region Selection ########################################
    def _promising_region_selection(self):
        # NOTE: the version in HiPA.py never fills X, so it crashes at `X[loc2idx[...]]`. Fixed here.
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
        best_idv = max(X, key=lambda idv: idv.adv_score.item())
        return best_idv, best_idv.patch

    def _fixed_region(self):
        best_idv = Individual(self.patch_s, (self.img_h, self.img_w))
        best_idv.location = tuple(self.predefined_loc)
        best_idv.patch = self.get_init_patch(self.patch_s)
        self.fitness.evaluate([best_idv])
        self._log([best_idv])
        self.prev_n_eval = self.fitness.n_eval
        return best_idv, best_idv.patch

    ############################################ Step-2: SignHunter ####################################################
    @staticmethod
    def sign_hunter(x0, loss_fn, eps, budget, stop_fn=None, lo=0., hi=1.):
        """
        Iterative SignHunter (maximisation), framework-agnostic so it can be unit-tested on CPU.
            x0      : tensor, starting point (any shape)
            loss_fn : callable(tensor) -> float, 1 call = 1 query
            budget  : max number of loss_fn calls
            stop_fn : callable(best_loss) -> bool, early stop (e.g. adv_score >= 0)
        Returns (x_best, f_best, n_queries, n_passes)
        """
        n = x0.numel()
        n_q = 0

        x_base = x0.clone()
        sgn = torch.ones(n, device=x0.device, dtype=x0.dtype)

        def make(s):
            return (x_base + eps * s.view_as(x_base)).clamp(lo, hi)

        x_best = make(sgn)
        f_best = loss_fn(x_best); n_q += 1
        h, i, n_passes = 0, 0, 0
        while n_q < budget and not (stop_fn is not None and stop_fn(f_best)):
            chunk = math.ceil(n / (2 ** h))
            start, end = i * chunk, min((i + 1) * chunk, n)

            sgn[start:end] *= -1
            x_new = make(sgn)
            if torch.equal(x_new, x_best):
                # every flipped coordinate is clipped at the box boundary -> nothing changes, save the query
                sgn[start:end] *= -1
            else:
                f_new = loss_fn(x_new); n_q += 1
                if f_new > f_best:
                    x_best, f_best = x_new, f_new
                else:
                    sgn[start:end] *= -1  # revert

            i += 1
            if i * chunk >= n:          # finished all chunks of this level
                h, i = h + 1, 0
                if chunk == 1:          # finished the finest level -> take the step, start a new pass
                    x_base = x_best.clone()
                    n_passes += 1
                    h = 0
                    x_cand = make(sgn)
                    if n_q < budget and not torch.equal(x_cand, x_best):
                        f_cand = loss_fn(x_cand); n_q += 1
                        if f_cand > f_best:
                            x_best, f_best = x_cand, f_cand
        return x_best, f_best, n_q, n_passes

    def _signHunter(self, best_idv, min_query, found):
        loc = best_idv.location
        if self.init == 'clean':
            x0 = self.fitness.img1[:, loc[0]:loc[1], loc[2]:loc[3]].clone()
        else:
            x0 = best_idv.patch.clone()

        state = {'best_idv': best_idv, 'min_query': min_query, 'found': found}

        def loss_fn(patch):
            idv = deepcopy(state['best_idv'])
            idv.patch = patch.clone()
            self.fitness.evaluate([idv])
            self._log([idv])
            self.prev_n_eval = self.fitness.n_eval
            if idv.adv_score >= 0 and not state['found']:
                state['found'], state['min_query'] = True, self.fitness.n_eval
            if idv.adv_score > state['best_idv'].adv_score:
                state['best_idv'] = idv
            return idv.adv_score.item()

        # keep 20 queries for Step-3 once an adversarial patch is found (same as HiPA)
        budget = self.max_query - self.fitness.n_eval - 20

        def stop_fn(f_best):
            return self.early_stop and f_best >= 0

        _, _, _, self.n_restarts = self.sign_hunter(x0, loss_fn, self.eps, budget, stop_fn)
        return state['best_idv'], state['min_query'], state['found']

    ####################################################### Main #######################################################
    def solve(self):
        from tqdm import tqdm
        found = False
        min_query = self.max_query

        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        # Step 1
        if self.predefined_loc is not None:
            best_idv, _ = self._fixed_region()
        else:
            best_idv, _ = self._promising_region_selection()
        if best_idv.adv_score >= 0:
            found, min_query = True, self.fitness.n_eval

        # Step 2: SignHunter
        if not (self.early_stop and found):
            best_idv, min_query, found = self._signHunter(best_idv, min_query, found)

        # Step 3: Stealth Refinement
        self.patch_before_refining = deepcopy(best_idv)
        if not self.early_stop:
            best_idv = self._refine(best_idv)
        return best_idv, best_idv.patch_size, min_query