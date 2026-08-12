import torch
import torch.nn.functional as F

class Fitness:
    def __init__(self, img1, img2, model, label, recons_w, attack_w, fitness_type, multi_objective=False):
        self.img1 = img1.cuda()
        self.img1_feature = model(img1.cuda().unsqueeze(0))
        self.img2_feature = model(img2.cuda().unsqueeze(0))
        self.model = model.eval()

        self.attack_w = attack_w
        self.recons_w = recons_w
        self.label = label
        self.fitness_type = fitness_type
        self.max_psnr, self.min_psnr = None, None
        self.max_adv, self.min_adv = None, None

        self.n_eval = 0

        self.multi_objective = multi_objective

    def init_self_check(self):
        with torch.no_grad():
            sims = F.cosine_similarity(self.img1_feature, self.img2_feature, dim=1)
            adv_scores = (1 - self.label) * (0.5 - sims) + self.label * (sims - 0.5)
            return adv_scores.item() >= 0.5

    def apply_patch_to_image(self, patch, location):
        img_copy = self.img1.clone()
        x_min, x_max, y_min, y_max = location
        img_copy[:, x_min:x_max, y_min:y_max] = patch
        return img_copy

    def evaluate_adv(self, list_imgs):
        with torch.no_grad():
            adv_batch = list_imgs.cuda()
            adv_features = self.model(adv_batch)
            sims = F.cosine_similarity(adv_features, self.img2_feature, dim=1)
            adv_scores = (1 - self.label) * (0.5 - sims) + self.label * (sims - 0.5)
            return adv_scores

    def evaluate_psnr(self, list_imgs):
        mse = F.mse_loss(list_imgs, self.img1.expand_as(list_imgs), reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1)
        psnr_scores = torch.log10(1 / (mse + 1e-8))
        return psnr_scores / 10

    def update_min_max(self, adv_scores, psnr_scores):
        self.min_psnr = torch.min(psnr_scores.min(), self.min_psnr)
        self.max_psnr = torch.max(psnr_scores.max(), self.max_psnr)
        self.min_adv = torch.min(adv_scores.min(), self.min_adv)
        self.max_adv = torch.max(psnr_scores.min(), self.min_adv)

    def evaluate(self, X):
        # In the original repo, the below is perform two times (one for evaluate_adv and one for evaluate_psnr)
        adv_imgs = torch.stack([self.apply_patch_to_image(idv.patch, idv.location) for idv in X])

        # One query = Evaluate Adv + Evaluate PSNR
        adv_scores = self.evaluate_adv(adv_imgs)
        psnr_scores = self.evaluate_psnr(adv_imgs)

        self.n_eval += len(X)

        if self.fitness_type == "adaptive":
            adv_scores = torch.where(adv_scores > 0, torch.tensor(0.0, device=adv_scores.device), adv_scores)

        if not self.multi_objective:
            F_scores = adv_scores * self.attack_w + psnr_scores * self.recons_w
        else:
            F_scores = [[adv_scores[i].cpu(), psnr_scores[i].cpu()] for i in range(len(adv_scores))]

        for i, idv in enumerate(X):
            idv.adv_score = adv_scores[i]
            idv.psnr_score = psnr_scores[i]
            idv.F = F_scores[i]
