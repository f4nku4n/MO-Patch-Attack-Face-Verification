
import math
from copy import deepcopy
from tqdm import tqdm

from core import Individual
from algorithm.IMS_HiPA_SignHunter import IMS_HiPA_SignHunter


class LexiKey:
    """Wraps the tuple (n_success, deficit, margin) so SignHunter (which compares its objective values with `>`
    and starts from f_best = -inf) compares patches on the tuple, without turning it into one number."""
    __slots__ = ('t',)

    def __init__(self, t):
        self.t = t

    def __gt__(self, other):
        if isinstance(other, LexiKey):
            return self.t > other.t
        return other == -math.inf

    def __lt__(self, other):
        if isinstance(other, LexiKey):
            return self.t < other.t
        return other == math.inf

    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        return not self.__gt__(other)

    def __eq__(self, other):
        return isinstance(other, LexiKey) and self.t == other.t

    def __repr__(self):
        return f'LexiKey{self.t}'


class IMS_HiPA_SignHunter_MultiRef(IMS_HiPA_SignHunter):

    @staticmethod
    def _better(a, b):
        """Is patch a better than patch b? (n_success, then deficit, then margin)"""
        return a.key > b.key

    ######################################## Step-1: Promising Region Selection ########################################
    def _promising_region_selection(self, patch_s):
        list_locs = self._get_all_locations(patch_s)
        best_idv, best_key = None, None
        for loc in list_locs:
            _X = [Individual(patch_s, (self.img_h, self.img_w)) for _ in range(self.n_warmup)]
            for idv in _X:
                idv.location = loc
            self.fitness.evaluate(_X)
            self._log()
            self.prev_n_eval = self.fitness.n_eval

            idx_best = 0
            for j in range(1, len(_X)):
                if self._better(_X[j], _X[idx_best]):
                    idx_best = j
            if best_key is None or _X[idx_best].key > best_key:  # score of the cell = key of its best patch
                best_idv, best_key = _X[idx_best], _X[idx_best].key
        return best_idv, best_idv.patch

    ######################################## Step-2: SignHunter ########################################
    def _sh_query(self, patch_s, patch):
        state = self.sh[patch_s]
        new_idv = deepcopy(state['idv'])
        new_idv.patch = patch.clone()

        self.fitness.evaluate([new_idv])
        self._log()
        self.prev_n_eval = self.fitness.n_eval

        if self._better(new_idv, state['idv']):
            state['idv'] = new_idv
        return LexiKey(new_idv.key)

    ############################################ Step-3: Stealth Refinement ############################################
    @staticmethod
    def _accept_refined(new_idv, best_idv, base_n):
        if new_idv.n_success > base_n:
            return True
        if new_idv.n_success < base_n:
            return False
        return bool(new_idv.success and new_idv.psnr_score > best_idv.psnr_score)

    def _refine(self, idv):
        if self.fitness.n_eval >= self.max_query:
            return idv
        base_n = idv.n_success
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

            if self._accept_refined(new_idv, best_idv, best_idv.n_success):  # must not lose what loop 1 gained
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
        """Same loop as IMS_HiPA_SignHunter.solve; comparisons use the key, success = all N fooled."""
        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval
        max_query = self.max_query - 20
        t = 1
        patch_s = 8
        min_query = self.max_query
        found = False
        list_best_idv, list_best_patch = [], []

        cannot_add_new_patch = False
        idv, patch = self._promising_region_selection(patch_s)
        if idv.success:
            found = True
            min_query = self.fitness.n_eval
        list_best_idv.append(idv)
        list_best_patch.append(patch)
        self.update_history(idv, patch_s)

        n_step = 1
        while (self.variant1 or not found) and self.fitness.n_eval < max_query:
            n_step += 1
            i = 0
            while i < len(list_best_idv):
                if n_step % (self.b ** i) == 0:
                    patch_size = list_best_idv[i].patch_size
                    list_best_idv[i], list_best_patch[i] = self._signHunter(list_best_idv[i], list_best_patch[i],
                                                                            patch_size)
                    self.update_history(list_best_idv[i], patch_size)
                    while i > 0 and self._better(list_best_idv[i], list_best_idv[i - 1]):
                        del list_best_idv[i - 1]
                        del list_best_patch[i - 1]
                        i -= 1
                    if list_best_idv[i].success:
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
                    cannot_add_new_patch = True
                    break
                else:
                    idv, patch = self._promising_region_selection(patch_s)
                    self.update_history(idv, patch_s)
                    list_best_idv.append(idv)
                    list_best_patch.append(patch)
                    if idv.success:
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
                    list_best_idv[i], list_best_patch[i] = self._signHunter(list_best_idv[i], list_best_patch[i],
                                                                            patch_size)
                    self.update_history(list_best_idv[i], patch_size)
                    while i > 0 and self._better(list_best_idv[i], list_best_idv[i - 1]):
                        del list_best_idv[i - 1]
                        del list_best_patch[i - 1]
                        i -= 1
                    if list_best_idv[i].success:
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
            list_best_idv[0], list_best_patch[0] = self._signHunter(list_best_idv[0], list_best_patch[0], patch_size)
            self.update_history(list_best_idv[0], patch_size)

        # step 3
        patch = list_best_idv[0]
        self.patch_before_refining = deepcopy(patch)
        patch = self._refine(patch)
        return patch, patch_size, min_query