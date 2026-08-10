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
    parser.add_argument('--model_dir', type=str, default='./pretrained_model')
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

    result_ssim, result_lpips = [], []
    for i in range(n_tested_imgs):
        log_path = f'{args.exp_log_dir}/{i}.p'

        data = p.load(open(log_path, 'rb'))
        try:
            cls_img = data["cls_img"]
        except:
            cls_img, _, _ = DATA[i]
            cls_img = cls_img.resize((img_height, img_width))
            cls_img = toTensor(cls_img).cpu().detach().numpy()
        adv_img = data["adv_img"]

        ssim_score = compute_SSIM(cls_img, adv_img)
        lpips_score = compute_LPIPS(cls_img, adv_img, lpips_model, device)

        result_ssim.append(ssim_score)
        result_lpips.append(lpips_score)

    mean_ssim = np.mean(result_ssim)
    mean_lpips = np.mean(result_lpips)

    if len(result_ssim) > 1:
        std_ssim = np.std(result_ssim)
        std_lpips = np.std(result_lpips)
    else:
        std_ssim = 0.0
        std_lpips = 0.0
    print(f'SSIM: {mean_ssim:.4f} ({std_ssim:.4f})')
    print(f'LPIPS: {mean_lpips:.4f} ({std_lpips:.4f})')