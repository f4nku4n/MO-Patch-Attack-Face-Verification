import math
import torch
import numpy as np
from tqdm import tqdm
from core import Individual

from algorithm.PatchRS import render, mutate

class CamoPatch:
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, temp, update_loc_period=4):
        self.max_query = max_query
        self.img_h, self.img_w = img_h, img_w
        self.patch_s = patch_s

        self.fitness = fitness

        self.temp = temp
        self.update_loc_period = update_loc_period

        self.N = 100
        self.mut = 0.5

        self.pbar = None
        self.prev_n_eval = 0

    def solve(self):
        found = False
        min_query = self.max_query

        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        h, w = self.img_h, self.img_w
        s = self.patch_s

        best_patch_geno = np.random.rand(self.N, 7)
        best_patch = render(best_patch_geno, s, s)
        best_loc = np.random.randint(h - s, size=2)

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

        patch_counter = 0
        for it in range(1, self.max_query):
            patch_counter += 1
            if patch_counter < self.update_loc_period:
                patch_new_geno = mutate(best_patch_geno, self.mut)
                patch_new = render(patch_new_geno, s, s)

                new_idv = Individual(self.patch_s, (self.img_h, self.img_w))
                new_idv.patch = torch.from_numpy(patch_new.copy()).permute(2, 0, 1).cuda()
                new_idv.location = (best_loc[0], best_loc[0] + s, best_loc[1], best_loc[1] + s)
                self.fitness.evaluate([new_idv])
                self.pbar.update(self.fitness.n_eval - self.prev_n_eval)
                self.pbar.set_postfix(query=self.fitness.n_eval)
                self.prev_n_eval = self.fitness.n_eval

                if new_idv.adv_score >= 0 and not found:
                    found = True
                    min_query = self.fitness.n_eval

                if best_idv.adv_score >= 0 and new_idv.adv_score >= 0:
                    if new_idv.psnr_score > best_idv.psnr_score:
                        best_patch_geno = patch_new_geno
                        best_patch = patch_new
                        best_idv = new_idv
                else:
                    if new_idv.adv_score > best_idv.adv_score:  # minimization
                        best_patch_geno = patch_new_geno
                        best_patch = patch_new
                        best_idv = new_idv

            else:
                patch_counter = 0

                # location update
                sh_i = int(max(sh_selection(self.max_query, it) * h, 0))
                sw_i = int(max(sh_selection(self.max_query, it) * w, 0))
                loc_new = best_loc.copy()
                loc_new = update_location(loc_new, sh_i, sw_i, h, w, s)

                new_idv = Individual(self.patch_s, (self.img_h, self.img_w))
                new_idv.patch = torch.from_numpy(best_patch.copy()).permute(2, 0, 1).cuda()
                new_idv.location = (loc_new[0], loc_new[0] + s, loc_new[1], loc_new[1] + s)

                self.fitness.evaluate([new_idv])
                self.pbar.update(self.fitness.n_eval - self.prev_n_eval)
                self.pbar.set_postfix(query=self.fitness.n_eval)
                self.prev_n_eval = self.fitness.n_eval

                if new_idv.adv_score >= 0 and not found:
                    found = True
                    min_query = self.fitness.n_eval

                if best_idv.adv_score >= 0 and new_idv.adv_score >= 0:
                    if new_idv.psnr_score > best_idv.psnr_score:
                        best_loc = loc_new
                        best_idv = new_idv

                else:
                    diff = new_idv.adv_score - best_idv.adv_score
                    curr_temp = self.temp / (it + 1)
                    metropolis = math.exp(-diff / curr_temp)

                    if new_idv.adv_score < best_idv.adv_score or np.random.rand() < metropolis:
                        best_loc = loc_new
                        best_idv = new_idv

        best_patch_size = best_idv.patch_size
        return best_idv, best_patch_size, min_query

###################################################### Utilities #######################################################
def compose_image(image, patch, location, clip_bounds):
    image_adv = image.copy()
    x, y = int(location[0]), int(location[1])
    patch_h, patch_w = patch.shape[:2]
    image_adv[x:x + patch_h, y:y + patch_w, :] = patch
    return np.clip(image_adv, clip_bounds[0], clip_bounds[1])


def update_location(loc_new, h_i, w_i, h, w, s):
    loc_new += np.random.randint(low=-h_i, high=h_i + 1, size=(2,))
    loc_new = np.clip(loc_new, 0, h - s)
    return loc_new

def sh_selection(n_queries, it):
    """ schedule to decrease the parameter p """
    t = max((float(n_queries - it) / n_queries - .0) ** 1., 0) * .75
    return t