import torch
import random
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy

from core import Individual
from utils.evolutionary_algorithms import isBetter

from torchvision import transforms

class IMS_HiPA_selected:
    def __init__(self, img_ref , max_query, img_h, img_w, fitness, b, variant1=False, variant2=False, n_warmup=1,
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

        self.img_ref = img_ref

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
    def random_int(low=0, high=1, shape=(1,)):
        t = low + (high - low) * torch.rand(shape).to('cuda')
        return t.long()

    def get_init_patch(self, s, n_iter=1000):
        patch_univ = torch.zeros([3, s, s]).to('cuda')
        for _ in range(n_iter):
            size_init = torch.randint(low=1, high=math.ceil(s ** .5), size=[1]).item()
            loc_init = torch.randint(s - size_init + 1, size=[2])
            patch_univ[
                :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = self.random_choice(
                [3, 1, 1])
        return patch_univ.clamp(0., 1.)

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

        toTensor = transforms.ToTensor()

        X = []
        for loc in list_locs:
            _X = [Individual(patch_s, (self.img_h, self.img_w)) for _ in range(self.n_warmup)]
            for idv in _X:
                idv.location = loc
                the_best_patch = None
                the_best_score = -2
                for l in list_locs:
                    x1, x2, y1, y2 = l
                    idv.patch = self.img_ref.crop((y1, x1, y2, x2))
                    idv.patch = toTensor(idv.patch).cuda()
                    self.fitness.evaluate([idv])
                    self._log()
                    self.prev_n_eval = self.fitness.n_eval
                    if idv.adv_score.item() > the_best_score:
                        the_best_patch = idv.patch
                        the_best_score = idv.adv_score.item()
                idv.patch = the_best_patch
                self.fitness.evaluate([idv])
            # self.fitness.evaluate(_X)
            # self._log()
            # self.prev_n_eval = self.fitness.n_eval

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
    def _hillClimbing(self, best_idv, best_patch, patch_s, min_query, found):
        max_query = self.max_query

        it = 0
        for it in range(self.max_query):
            s_it = int(max(self.p_selection(it) ** .5 * patch_s, 1))
            if s_it == 1:
                break
        it_start_cu = it + (self.max_query - it) // 2

        while self.fitness.n_eval < max_query:
            s_it = int(max(self.p_selection(it) ** .5 * patch_s, 1))
            p_it = torch.randint(patch_s - s_it + 1, size=[2])

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
            self._log()
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
        return best_idv, best_patch, min_query, found

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
        t = 1
        patch_s = 20
        min_query = self.max_query
        found = False
        list_best_idv, list_best_patch = [], []

        cannot_add_new_patch = False
        # Init the first HiPA
        idv, patch = self._promising_region_selection(patch_s)
        if idv.adv_score >= 0:
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
                    list_best_idv[i], list_best_patch[i], min_query, found = self._hillClimbing(list_best_idv[i], list_best_patch[i],
                                                                              patch_size, min_query, found)
                    self.update_history(list_best_idv[i], patch_size)

                    while i > 0 and list_best_idv[i].adv_score > list_best_idv[i - 1].adv_score:
                        # print(f'Kill HiPA (p={list_best_idv[i - 1].patch_size})')
                        del list_best_idv[i - 1]
                        del list_best_patch[i - 1]
                        i -= 1

                    if list_best_idv[i].adv_score >= 0:
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
                query = math.ceil(self.img_h / patch_s)**2 * math.ceil(self.img_w / patch_s)**2 * self.n_warmup
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
                    if idv.adv_score >= 0:
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
                    list_best_idv[i], list_best_patch[i], min_query, found = self._hillClimbing(list_best_idv[i], list_best_patch[i],
                                                                              patch_size, min_query, found)
                    self.update_history(list_best_idv[i], patch_size)
                    while i > 0 and list_best_idv[i].adv_score > list_best_idv[i - 1].adv_score:
                        del list_best_idv[i - 1]
                        del list_best_patch[i - 1]
                        i -= 1
                    if list_best_idv[i].adv_score >= 0:
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
            list_best_idv[0], list_best_patch[0], min_query, found = self._hillClimbing(list_best_idv[0], list_best_patch[0], patch_size, min_query, found)
            self.update_history(list_best_idv[0], patch_size)

        # step 3
        patch = list_best_idv[0]
        self.patch_before_refining = deepcopy(patch)
        patch = self._refine(patch)
        return patch, patch_size, min_query

