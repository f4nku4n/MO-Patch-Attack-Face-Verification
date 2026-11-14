import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from copy import deepcopy
from core import Individual


class iFGSM:
    def __init__(self, max_query, img_h, img_w, patch_s, fitness, eps=32 / 255, alpha=32 / 255, predefined_loc=None):
        self.max_query = max_query
        self.img_h, self.img_w = img_h, img_w
        self.patch_s = patch_s

        self.fitness = fitness

        self.loc = predefined_loc
        self.adv_score_start = None
        self.eps, self.alpha = eps, alpha

    def _get_all_locations(self):
        list_locs = []
        for i in range(0, self.img_h - self.patch_s + 1, self.patch_s):
            for j in range(0, self.img_w - self.patch_s + 1, self.patch_s):
                list_locs.append((i, i + self.patch_s, j, j + self.patch_s))
        return list_locs

    def _promising_region_selection(self):
        list_locs = self._get_all_locations()
        results = {loc: {} for loc in list_locs}
        loc2idx = {loc: i for i, loc in enumerate(list_locs)}

        X = []
        idv = Individual(self.patch_s, (self.img_h, self.img_w))
        for loc in list_locs:
            _idv = deepcopy(idv)
            _idv.location = loc
            X.append(_idv)
        self.fitness.evaluate(X)

        for idv in X:
            loc = idv.location
            results[loc] = idv.adv_score
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        list_sorted_locs = list(results.keys())
        print(list_sorted_locs)
        loc_best = list_sorted_locs[0]
        best_idv = X[loc2idx[loc_best]]
        best_patch = best_idv.patch

        return best_idv, best_patch

    @torch.enable_grad()
    def solve(self):
        """Optimize a fixed patch and return its detached CHW pixels."""
        if self.loc is None:
            best_idv, _ = self._promising_region_selection()
            self.loc = best_idv.location
        x_min, x_max, y_min, y_max = self.loc
        region = (slice(None), slice(x_min, x_max), slice(y_min, y_max))

        reference = self.fitness.img2_feature.detach()
        image = self.fitness.img1.detach().to(reference.device)

        patch0 = image[region].detach().clone()
        patch = patch0.clone()

        pbar = tqdm(total=self.max_query, desc="PGD", unit="iter")

        for _ in range(self.max_query):
            patch.requires_grad_()
            adv_img = image.clone()
            adv_img[region] = patch
            adv_batch = adv_img.unsqueeze(0)

            features = self.fitness.model(adv_batch)
            self.fitness.n_eval += 1
            similarity = F.cosine_similarity(features, reference, dim=1)
            attack_score = (1 - self.fitness.label) * (self.fitness.threshold - similarity)
            attack_score = attack_score + self.fitness.label * (similarity - self.fitness.threshold)
            pbar.set_postfix(adv_score=f"{attack_score.item():.4f}")
            pbar.update(1)

            if self.adv_score_start is None:
                self.adv_score_start = attack_score.item()

            if attack_score.item() >= 0:
                break
            (gradient,) = torch.autograd.grad(attack_score, patch)
            with torch.no_grad():
                # patch = patch + alpha * gradient.sign()
                # patch = torch.max(torch.min(patch, patch0 + eps), patch0 - eps)
                # patch = patch.clamp(0, 1).detach()
                patch = (patch + gradient.sign() * self.alpha).clamp(0, 1)
        pbar.close()
        best_patch = Individual(patch_size=self.patch_s, img_shape=(self.img_h, self.img_w))
        best_patch.patch = patch.detach()
        best_patch.location = self.loc
        self.fitness.evaluate([best_patch])
        self.fitness.n_eval -= 1
        return best_patch, None, self.fitness.n_eval
