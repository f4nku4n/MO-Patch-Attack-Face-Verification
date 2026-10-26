import argparse
import numpy as np
import pickle as p

import torch
from torchvision import transforms


from dataset import LFW


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tested_imgs', type=int, default=100, help="the number of tested images")
    parser.add_argument('--victim_model_name', type=str, default='vggface',
                        choices=['vggface', 'webface', 'arcface', 'cosface'],
                        help='pretrained victim model')
    parser.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    parser.add_argument('--mask_dir', type=str, default='./mask')
    parser.add_argument('--exp_log_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_tested_imgs = args.n_tested_imgs
    victim_model_name = args.victim_model_name
    if victim_model_name in ['vggface', 'webface']:
        img_height, img_width = 160, 160
    else:  # arcface, cosface
        img_height, img_width = 112, 112

    pair_path = './lfw_preprocess/pairs.txt'
    if victim_model_name != 'vggface':
        pair_path = f'./lfw_preprocess/{n_tested_imgs}pairs_{victim_model_name}.txt'

    DATA = LFW(IMG_DIR=args.img_dir, MASK_DIR=args.mask_dir, PAIR_PATH=pair_path, transform=None)
    toTensor = transforms.ToTensor()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    success_list, psnr_list = [], []
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

        success_list.append(data['success_attack'])
        psnr_list.append(data['psnr_score'])

    mean_psnr = np.mean(psnr_list)

    if len(psnr_list) > 0:
        std_psnr = np.std(psnr_list)
    else:
        std_psnr=0.0
    print(f"Success rate: {sum(success_list) / len(success_list):.4f}")
    print(f'PSNR: {mean_psnr:.4f} ({std_psnr:.4f})')
