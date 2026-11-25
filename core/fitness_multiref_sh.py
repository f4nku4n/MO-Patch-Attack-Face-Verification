import torch
import torch.nn.functional as F


class MultiRefFitnessSH:
    def __init__(self, img1, ref_imgs, model, threshold, batch_size=64):
        self.model = model.eval()
        self.img1 = img1.cuda()
        self.threshold = threshold
        self.label = 0  # same identity -> dodging
        self.n_eval = 0
        with torch.no_grad():
            feats = [model(torch.stack(ref_imgs[i:i + batch_size]).cuda())
                     for i in range(0, len(ref_imgs), batch_size)]
            self.ref_feats = F.normalize(torch.cat(feats), dim=1)  # (N, d)
            self.img1_feature = model(self.img1.unsqueeze(0))
        self.n_refs = self.ref_feats.shape[0]

    def apply_patch_to_image(self, patch, location):
        img_copy = self.img1.clone()
        x_min, x_max, y_min, y_max = location
        img_copy[:, x_min:x_max, y_min:y_max] = patch
        return img_copy

    @torch.no_grad()
    def adv_matrix(self, imgs):
        """imgs: (B, 3, H, W) -> (B, N) adv scores against every reference"""
        feats = F.normalize(self.model(imgs.cuda()), dim=1)
        return self.threshold - feats @ self.ref_feats.T

    def init_self_check(self):
        """Clean probe must be verified as the same person by every reference."""
        return bool((self.adv_matrix(self.img1.unsqueeze(0)) < 0).all())

    def evaluate_psnr(self, list_imgs):
        # same scale as core/fitness.py on this branch (dB)
        mse = F.mse_loss(list_imgs, self.img1.expand_as(list_imgs), reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1)
        return torch.log10(1 / (mse + 1e-8)) * 10

    def evaluate(self, X):
        adv_imgs = torch.stack([self.apply_patch_to_image(idv.patch, idv.location) for idv in X])
        adv_mat = self.adv_matrix(adv_imgs)  # one forward pass per individual = one query each
        psnr = self.evaluate_psnr(adv_imgs)
        self.n_eval += len(X)
        for i, idv in enumerate(X):
            vec = adv_mat[i]
            idv.adv_vec = vec
            idv.n_success = int((vec >= 0).sum())
            idv.success = idv.n_success == self.n_refs
            idv.deficit = float(vec.clamp(max=0).sum())
            idv.margin = float(vec.min())
            idv.key = (idv.n_success, idv.deficit, idv.margin)
            idv.adv_score = vec.min()
            idv.psnr_score = psnr[i]
            idv.F = idv.adv_score