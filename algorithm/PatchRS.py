import torch
import math
import numpy as np
from tqdm import tqdm

from core import Individual

class PatchRS:
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, p_init=0.8, update_loc_period=4):
        self.max_query = max_query
        self.img_h, self.img_w = img_h, img_w
        self.patch_s = patch_s

        self.fitness = fitness

        self.pbar = None
        self.prev_n_eval = 0

        self.p_init = p_init

        self.update_loc_period = update_loc_period
        self.rescale_schedule = True

    def p_selection(self, it):
        """ schedule to decrease the parameter p """

        if self.rescale_schedule:
            it = int(it / self.max_query * 10000)

        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init

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

    def solve(self):
        found = False
        min_query = self.max_query

        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        H, W = self.img_h, self.img_w
        s = self.patch_s

        best_loc = [np.random.randint(0, H - s), np.random.randint(0, W - s)]
        best_patch = self.get_init_patch(s)

        loc_t = abs(self.update_loc_period)
        assert loc_t > 1

        best_idv = Individual(self.patch_s, (self.img_h, self.img_w))
        best_idv.patch = best_patch.clone()
        best_idv.location = (best_loc[0], best_loc[0] + s, best_loc[1], best_loc[1] + s)

        self.fitness.evaluate([best_idv])
        self.pbar.update(self.fitness.n_eval - self.prev_n_eval)
        self.pbar.set_postfix(query=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        if best_idv.adv_score >= 0 and not found:
            found = True
            min_query = self.fitness.n_eval
        if not found:
            # set when to start single channel updates
            it = 0
            for it in range(self.max_query):
                s_it = int(max(self.p_selection(it) ** .5 * s, 1))
                if s_it == 1:
                    break
            it_start_cu = it + (self.max_query - it) // 2

            for it in range(1, self.max_query):
                s_it = int(max(self.p_selection(it) ** .5 * s, 1))
                p_it = torch.randint(s - s_it + 1, size=[2])

                ## Sample update
                new_patch = best_patch.clone()
                new_loc = best_loc.copy()

                sh_it = int(max(self.sh_selection(it) * H, 0))
                sw_it = int(max(self.sh_selection(it) * W, 0))
                update_loc = (it % loc_t == 0) and (sh_it > 0) and (sw_it > 0)
                if self.update_loc_period < 0 < sh_it and sw_it > 0:
                    update_loc = not update_loc
                update_patch = not update_loc

                if update_patch:
                    # update patch
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

                if update_loc:
                    new_loc[0] = np.clip(new_loc[0] + np.random.randint(-sh_it, sh_it + 1), 0, H - s)
                    new_loc[1] = np.clip(new_loc[1] + np.random.randint(-sw_it, sw_it + 1), 0, W - s)

                new_idv = Individual(self.patch_s, (self.img_h, self.img_w))
                new_idv.patch = new_patch.clone()
                new_idv.location = (new_loc[0], new_loc[0] + s, new_loc[1], new_loc[1] + s)
                self.fitness.evaluate([new_idv])
                self.pbar.update(self.fitness.n_eval - self.prev_n_eval)
                self.pbar.set_postfix(query=self.fitness.n_eval)
                self.prev_n_eval = self.fitness.n_eval

                if new_idv.adv_score > best_idv.adv_score:
                    best_patch = new_patch.clone()
                    best_loc = new_loc.copy()
                    best_idv = new_idv

                if best_idv.adv_score >= 0:
                    min_query = self.fitness.n_eval
                    break

        best_patch_size = best_idv.patch_size
        return best_idv, best_patch_size, min_query
