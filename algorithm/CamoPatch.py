import cv2
import math
import torch
import numpy as np
from tqdm import tqdm
from core import Individual


class CamoPatch:
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, mut=0.3, temp=300, update_loc_period=4):
        self.max_query = max_query
        self.img_h, self.img_w = img_h, img_w
        self.patch_s = patch_s

        self.fitness = fitness

        self.temp = temp
        self.update_loc_period = update_loc_period

        self.N = 100
        self.mut = mut

        self.pbar = None
        self.prev_n_eval = 0

    def solve(self):
        found = False
        min_query = self.max_query

        self.pbar = tqdm(total=self.max_query, initial=self.fitness.n_eval)
        self.prev_n_eval = self.fitness.n_eval

        h, w = self.img_h, self.img_w
        s = self.patch_s

        # Best patch
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
            if patch_counter < self.update_loc_period:  # Update patch content
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
                        best_patch_geno = patch_new_geno.copy()
                        best_patch = patch_new.copy()
                        best_idv = new_idv
                else:
                    if new_idv.adv_score > best_idv.adv_score:
                        best_patch_geno = patch_new_geno.copy()
                        best_patch = patch_new.copy()
                        best_idv = new_idv

            else:  # Update location
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
                    if best_idv.adv_score < 0:
                        diff = new_idv.adv_score - best_idv.adv_score
                        curr_temp = self.temp / (it + 1)
                        metropolis = math.exp(-diff / curr_temp)

                        if new_idv.adv_score > best_idv.adv_score or np.random.rand() < metropolis:
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
