import argparse
import numpy as np
import pickle as p

import torch
from torchvision import transforms

import lpips
from skimage.metrics import structural_similarity as ssim

from dataset import LFW

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tested_imgs', type=int, default=100, help="the number of tested images")

    parser.add_argument('--pair_path', type=str, default='lfw_preprocess/pairs.txt')
    parser.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    parser.add_argument('--mask_dir', type=str, default='./mask')
    parser.add_argument('--exp_log_dir', type=str, required=True)
    return parser.parse_args()


def compute_LPIPS(original, adversarial, lpips_model, device):
    if isinstance(original, np.ndarray):
        original = torch.from_numpy(original).float()
    if isinstance(adversarial, np.ndarray):
        adversarial = torch.from_numpy(adversarial).float()

    original = original.unsqueeze(0).to(device)
    adversarial = adversarial.unsqueeze(0).to(device)

    original = original * 2.0 - 1.0
    adversarial = adversarial * 2.0 - 1.0

    with torch.no_grad():
        score = lpips_model(original, adversarial)

    return float(score.item())


def compute_SSIM(original, adversarial):
    if not isinstance(original, np.ndarray):
        original = original.cpu().detach().numpy()
    if not isinstance(adversarial, np.ndarray):
        adversarial = adversarial.cpu().detach().numpy()
    score = ssim(original, adversarial, data_range=1.0, channel_axis=0)
    return float(score)


if __name__ == "__main__":
    args = parse_args()
    DATA = LFW(IMG_DIR=args.img_dir, MASK_DIR=args.mask_dir, PAIR_PATH=args.pair_path, transform=None)
    toTensor = transforms.ToTensor()
    n_tested_imgs = args.n_tested_imgs
    img_height, img_width = 160, 160

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    success_list, psnr_list, ssim_list, lpips_list = [], [], [], []
    for i in range(n_tested_imgs):
        log_path = f'{args.exp_log_dir}/{i}.p'

        data = p.load(open(log_path, 'rb'))
        try:
            cls_img = data["cls_img"]
        except KeyError:
            cls_img, _, _ = DATA[i]
            cls_img = cls_img.resize((img_height, img_width))
            cls_img = toTensor(cls_img).cpu().detach().numpy()
        adv_img = data["adv_img"]

        ssim_score = compute_SSIM(cls_img, adv_img)
        lpips_score = compute_LPIPS(cls_img, adv_img, lpips_model, device)

        success_list.append(data['success_attack'])
        psnr_list.append(data['psnr_score'])
        ssim_list.append(ssim_score)
        lpips_list.append(lpips_score)

    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    mean_lpips = np.mean(lpips_list)

    if len(psnr_list) > 0:
        std_psnr = np.std(psnr_list)
        std_ssim = np.std(ssim_list)
        std_lpips = np.std(lpips_list)
    else:
        std_psnr, std_ssim, std_lpips = 0.0, 0.0, 0.0
    print(f"Success rate: {sum(success_list) / len(success_list):.4f}")
    print(f'PSNR: {mean_psnr:.4f} ({std_psnr:.4f})')
    print(f'SSIM: {mean_ssim:.4f} ({std_ssim:.4f})')
    print(f'LPIPS: {mean_lpips:.4f} ({std_lpips:.4f})')