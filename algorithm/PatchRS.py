import cv2
import torch
import numpy as np
from tqdm import tqdm

from core import Individual

def render(x, h, w):
    phenotype = np.ones((h, w, 3)) # load a white patch w*w
    radius_avg = (phenotype.shape[0] + phenotype.shape[1]) / 2 / 6
    for row in x:
        overlay = phenotype.copy()
        cv2.circle(
            overlay,
            center=(int(row[1] * h), int(row[0] * w)),
            radius=int(row[2] * radius_avg),
            color=(int(row[3] * 255), int(row[4] * 255), int(row[5] * 255)),
            thickness=-1,
        )
        alpha = row[6]
        phenotype = cv2.addWeighted(overlay, alpha, phenotype, 1 - alpha, 0)
    return phenotype / 255.0

def mutate(soln, mut=0.3):
    new_specie = soln.copy()

    # Randomization for Evolution
    genes = soln.shape[0]
    length = soln.shape[1]
    y = np.random.randint(0, genes)
    change = np.random.randint(0, length + 1)

    selection = np.random.choice(length, size=change, replace=False)

    if np.random.rand() < mut:
        new_specie[y, selection] = np.random.rand(len(selection))
    else:
        new_specie[y, selection] += (np.random.rand(len(selection)) - 0.5) / 3
        new_specie[y, selection] = np.clip(new_specie[y, selection], 0, 1)
    return new_specie

class PatchRS:
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, p_init=0.4, update_loc_period=4):
        self.max_query = max_query
        self.img_h, self.img_w = img_h, img_w
        self.patch_s = patch_s

        self.fitness = fitness

        self.pbar = None
        self.prev_n_eval = 0

        self.p_init = p_init

        self.N = 100
        self.update_loc_period = update_loc_period
        self.rescale_schedule = True

        self.init_patches = 'random_squares'

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

    def get_init_patch(self, s):
        patch_geno = np.random.rand(self.N, 7)  # Difference
        patch = render(patch_geno, s, s)
        return patch_geno, patch

    def solve(self):
        found = False
        min_query = self.max_query

        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        H, W = self.img_h, self.img_w
        s = self.patch_s

        best_loc = [np.random.randint(0, H - s), np.random.randint(0, W - s)]
        best_patch_geno, best_patch = self.get_init_patch(s)

        loc_t = abs(self.update_loc_period)
        assert loc_t > 1

        best_idv = Individual(self.patch_s, (self.img_h, self.img_w))
        best_idv.patch = torch.from_numpy(best_patch.copy()).permute(2, 0, 1).cuda()
        best_idv.location = (best_loc[0], best_loc[0] + s, best_loc[1], best_loc[1] + s)

        self.fitness.evaluate([best_idv])
        self.pbar.update(self.fitness.n_eval - self.prev_n_eval)
        self.pbar.set_postfix(query=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        if best_idv.adv_score >= 0 and not found:
            found = True
            min_query = self.fitness.n_eval
        if not found:
            for it in range(1, self.max_query):
                s_it = int(max(self.p_selection(it) ** .5 * s, 1))

                ## Sample update
                new_patch_geno = best_patch_geno.copy()
                new_patch = best_patch.copy()
                new_loc = best_loc.copy()

                sh_it = int(max(self.sh_selection(it) * H, 0))
                sw_it = int(max(self.sh_selection(it) * W, 0))
                update_loc = (it % loc_t == 0) and (sh_it > 0) and (sw_it > 0)
                if self.update_loc_period < 0 < sh_it and sw_it > 0:
                    update_loc = not update_loc
                update_patch = not update_loc

                if update_patch:
                    # update patch
                    if s_it > 1:
                        new_patch_geno = mutate(best_patch_geno, mut=1.1)
                    else:
                        new_patch_geno = mutate(best_patch_geno, mut=-0.1)
                    new_patch = render(new_patch_geno, s, s)

                if update_loc:
                    new_loc[0] = np.clip(new_loc[0] + np.random.randint(-sh_it, sh_it + 1), 0, H - s)
                    new_loc[1] = np.clip(new_loc[1] + np.random.randint(-sw_it, sw_it + 1), 0, W - s)

                new_idv = Individual(self.patch_s, (self.img_h, self.img_w))
                new_idv.patch = torch.from_numpy(new_patch.copy()).permute(2, 0, 1).cuda()
                new_idv.location = (new_loc[0], new_loc[0] + s, new_loc[1], new_loc[1] + s)
                self.fitness.evaluate([new_idv])
                self.pbar.update(self.fitness.n_eval - self.prev_n_eval)
                self.pbar.set_postfix(query=self.fitness.n_eval)
                self.prev_n_eval = self.fitness.n_eval

                if new_idv.adv_score > best_idv.adv_score:
                    best_patch_geno = new_patch_geno.copy()
                    best_patch = new_patch.copy()
                    best_loc = new_loc.copy()
                    best_idv = new_idv

                if best_idv.adv_score >= 0:
                    min_query = self.fitness.n_eval
                    break

        best_patch_size = best_idv.patch_size
        return best_idv, best_patch_size, min_query
