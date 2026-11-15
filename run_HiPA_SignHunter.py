import os
import json
import argparse
import pickle as p

from core import Fitness
from algorithm.HiPA_SignHunter import HiPA_SignHunter

from dataset import LFW
from factory import get_model

from torchvision import transforms
from torchvision.utils import save_image

from utils.common import set_seed, NumpyEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_query', type=int, default=10000, help="Maximum number of evaluations")
    parser.add_argument('--n_warmup', type=int, default=1)
    parser.add_argument('--early_stop', action='store_true', help='Stop as soon as the attack succeeds')

    # SignHunter
    parser.add_argument('--eps', type=int, default=32, help='step size (x/255), same meaning as --alpha of run_iFGSM')
    parser.add_argument('--init', type=str, default='clean', choices=['clean', 'step1'],
                        help="clean: start from the clean pixels (like iFGSM); step1: start from Step-1's random patch")
    parser.add_argument('--patch_s', type=int, default=None, help='override the default patch size')
    parser.add_argument('--ifgsm_log_dir', type=str, default=None,
                        help="<iFGSM exp_dir>/logs -> reuse iFGSM's location for each image (fair comparison)")
    parser.add_argument('--img_ids', type=int, nargs='+', default=None,
                        help='only attack these image indices (0-based, same index as the .p files)')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--victim_model_name', type=str, default='vggface',
                        choices=['vggface', 'webface', 'arcface', 'cosface'])
    parser.add_argument('--n_tested_imgs', type=int, default=100)
    parser.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    parser.add_argument('--model_dir', type=str, default='./pretrained_model')
    parser.add_argument('--mask_dir', type=str, default='./mask')
    parser.add_argument('--exp_dir', type=str, default='./exp_results')
    parser.add_argument('--setting', type=str, default='same', choices=['same', 'different'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    victim_model_name = args.victim_model_name
    MODEL = get_model(model_name=victim_model_name, model_dir=args.model_dir)
    print('Load Pre-trained model - Done!')
    if victim_model_name in ['webface', 'vggface']:
        threshold, img_h, img_w, patch_s = 0.5, 160, 160, 20
    elif victim_model_name == 'cosface':
        threshold, img_h, img_w, patch_s = 0.30, 112, 112, 14
    elif victim_model_name == 'arcface':
        threshold, img_h, img_w, patch_s = 0.15, 112, 112, 14
    else:
        raise ValueError
    if args.patch_s is not None:
        patch_s = args.patch_s

    config = vars(args).copy()
    config.update({'method': 'HiPA_SignHunter', 'patch_s': patch_s, 'threshold': threshold})

    exp_dir = (f'{args.exp_dir}/HiPA_SignHunter_maxQuery-{args.max_query}_Setting-{args.setting}_'
               f'VictimModel-{victim_model_name}_PatchS-{patch_s}_Eps-{args.eps}_Init-{args.init}_'
               f'iFGSMLoc-{args.ifgsm_log_dir is not None}/Seed{args.seed}')
    exp_img_dir, exp_log_dir = f"{exp_dir}/images", f"{exp_dir}/logs"
    os.makedirs(exp_img_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)
    with open(f'{exp_dir}/configs.json', "w") as file:
        json.dump(config, file, indent=4, cls=NumpyEncoder)

    pair_path = f'./lfw_preprocess/{args.setting}_seed_{args.seed}.txt'
    DATA = LFW(IMG_DIR=args.img_dir, MASK_DIR=args.mask_dir, PAIR_PATH=pair_path, transform=None)
    print('Load Data - Done!')
    toTensor = transforms.ToTensor()

    img_ids = args.img_ids if args.img_ids is not None else list(range(args.n_tested_imgs))
    success_list = []
    for i in img_ids:
        print(f"Image #{i}")
        set_seed(args.seed)

        img1, img2, label = DATA[i]
        img1, img2 = img1.resize((img_h, img_w)), img2.resize((img_h, img_w))
        img1_torch, img2_torch = toTensor(img1), toTensor(img2)

        fitness = Fitness(img1=img1_torch, img2=img2_torch, model=MODEL, label=label,
                          recons_w=0.0, attack_w=0.0, fitness_type=None, threshold=threshold)
        if not fitness.init_self_check():
            print(f'Image #{i} - Model gives wrong prediction at initialization!')

        loc = None
        if args.ifgsm_log_dir is not None:
            ifgsm_res = p.load(open(f'{args.ifgsm_log_dir}/{i}.p', 'rb'))
            loc = tuple(int(v) for v in ifgsm_res['loc'])
            if loc[1] - loc[0] != patch_s:
                print(f'[WARN] iFGSM patch size = {loc[1] - loc[0]} != patch_s = {patch_s}. Use --patch_s {loc[1] - loc[0]}')
            print(f'Reuse iFGSM location {loc} (iFGSM success = {ifgsm_res["success_attack"]})')

        algo = HiPA_SignHunter(max_query=args.max_query, img_h=img_h, img_w=img_w, patch_s=patch_s,
                               fitness=fitness, n_warmup=args.n_warmup, early_stop=args.early_stop,
                               eps=args.eps / 255, init=args.init, predefined_loc=loc)
        best_patch, best_patch_size, min_query = algo.solve()
        algo.pbar.close()

        patch, loc = best_patch.patch, best_patch.location
        adv_img = fitness.apply_patch_to_image(patch, loc)
        adv_score, psnr_score = best_patch.adv_score.item(), best_patch.psnr_score.item()
        success_attack = (adv_score >= 0)
        success_list.append(success_attack)

        print(f"Adv Score: {adv_score:.4f} | PSNR: {psnr_score:.4f} | Success: {success_attack} | "
              f"Min Query: {min_query} | Loc: {loc} | SignHunter passes: {algo.n_restarts}")
        print('-' * 20)

        save_image(adv_img, f"{exp_img_dir}/{i}_Success-{success_attack}_AdvScore-{adv_score:.2f}_PSNRScore-{psnr_score:.2f}.png")
        results = {
            "loc": loc,
            "patch": patch.cpu().detach().numpy(),
            "adv_img": adv_img.cpu().detach().numpy(),
            "success_attack": success_attack,
            "adv_score": adv_score,
            "psnr_score": psnr_score,
            "min_query": min_query,
            "list_adv_psnr_scores": algo.history,
            "n_passes": algo.n_restarts,
        }
        p.dump(results, open(f'{exp_log_dir}/{i}.p', 'wb'))

    print(f"Success rate: {sum(success_list) / len(success_list)}")