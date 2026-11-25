from copy import deepcopy

from algorithm.IMS_HiPA_SignHunter import IMS_HiPA_SignHunter


class IMS_HiPA_SignHunter_MultiRef(IMS_HiPA_SignHunter):

    @staticmethod
    def _accept_refined(new_idv, best_idv, base_n):
        if new_idv.n_success > base_n:
            return True
        if new_idv.n_success < base_n:
            return False
        return bool(new_idv.success and new_idv.psnr_score > best_idv.psnr_score)

    ############################################ Step-3: Stealth Refinement ############################################
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