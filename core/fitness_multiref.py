import torch
import torch.nn.functional as F

AGGREGATIONS = ['max', 'min', 'mean', 'count', 'lexi']


def aggregate(adv_mat, agg):
    """adv_mat: (B, N) -> (B,)"""
    n = adv_mat.shape[1]
    n_succ = (adv_mat >= 0).sum(dim=1).float()
    if agg == 'max':
        return adv_mat.max(dim=1).values
    if agg == 'min':
        return adv_mat.min(dim=1).values
    if agg == 'mean':
        return adv_mat.mean(dim=1)
    if agg == 'count':
        return n_succ + adv_mat.mean(dim=1) / 4.0
    if agg == 'lexi':
        deficit = adv_mat.clamp(max=0).sum(dim=1)
        partial = n_succ + 1.0 / (1.0 - deficit)
        full = n + 1.0 + adv_mat.min(dim=1).values
        return torch.where(n_succ >= n, full, partial)
    raise ValueError(agg)


class MultiRefFitness:
    def __init__(self, img1, ref_imgs, model, agg='max', batch_size=64):
        self.model = model.eval()
        self.img1 = img1.cuda()
        self.agg = agg
        self.label = 0  # same identity -> dodging
        self.n_eval = 0
        with torch.no_grad():
            feats = [model(torch.stack(ref_imgs[i:i + batch_size]).cuda())
                     for i in range(0, len(ref_imgs), batch_size)]
            self.ref_feats = F.normalize(torch.cat(feats), dim=1)  # (N, d)
            self.img1_feature = model(self.img1.unsqueeze(0))
        self.n_refs = self.ref_feats.shape[0]

    # ---- helpers -------------------------------------------------------------------------------------
    def apply_patch_to_image(self, patch, location):
        img_copy = self.img1.clone()
        x_min, x_max, y_min, y_max = location
        img_copy[:, x_min:x_max, y_min:y_max] = patch
        return img_copy

    @torch.no_grad()
    def adv_matrix(self, imgs):
        """imgs: (B, 3, H, W) -> (B, N) adv scores against every reference"""
        feats = F.normalize(self.model(imgs.cuda()), dim=1)
        return 0.5 - feats @ self.ref_feats.T

    def init_self_check(self):
        """Clean probe must be verified as the same person by every reference."""
        return bool((self.adv_matrix(self.img1.unsqueeze(0)) < 0).all())

    def evaluate_psnr(self, list_imgs):
        mse = F.mse_loss(list_imgs, self.img1.expand_as(list_imgs), reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1)
        return torch.log10(1 / (mse + 1e-8)) / 10

    # ---- main entry used by the optimiser -----------------------------------------------------------
    def evaluate(self, X):
        adv_imgs = torch.stack([self.apply_patch_to_image(idv.patch, idv.location) for idv in X])
        adv_mat = self.adv_matrix(adv_imgs)          # one forward pass per individual = one query each
        scores = aggregate(adv_mat, self.agg)
        psnr = self.evaluate_psnr(adv_imgs)
        self.n_eval += len(X)
        for i, idv in enumerate(X):
            idv.adv_vec = adv_mat[i]
            idv.n_success = int((adv_mat[i] >= 0).sum())
            idv.success = idv.n_success == self.n_refs
            idv.adv_score = scores[i]
            idv.psnr_score = psnr[i]
            idv.F = scores[i]