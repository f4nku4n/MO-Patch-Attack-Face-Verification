import os
import json
import argparse
import numpy as np
import pickle as p

from fitness import Fitness
from algorithm import HillClimbing

from dataset import LFW
from factory import get_model

from torchvision import transforms
from torchvision.utils import save_image

from utils.common import set_seed, NumpyEncoder, checkSameConfigs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=20, help="Size of the patch")
    parser.add_argument('--max_query', type=int, default=10000, help="Maximum number of evaluations")

    parser.add_argument('--n_warmup', type=int, default=1)
    parser.add_argument('--step1_random', action='store_true', help='Step 1: Random Location')
    parser.add_argument('--step2_random', action='store_true', help='Step 2: Random Search')
    parser.add_argument('--early_stop', action='store_true', help='Early stop if all individual are the same')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--victim_model_name', type=str, default='vggface',
                        choices=['vggface', 'webface', 'arcface', 'cosface'],
                        help='pretrained victim model')
    parser.add_argument('--n_tested_imgs', type=int, default=100, help="the number of tested images")

    parser.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    parser.add_argument('--model_dir', type=str, default='./pretrained_model')
    parser.add_argument('--mask_dir', type=str, default='./mask')
    parser.add_argument('--exp_dir', type=str, default='./exp_results')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Save configurations
    config = {
        'method': 'Hill-Climbing',
        'patch_size': args.patch_size,
        'max_query': args.max_query,
        'n_warmup': args.n_warmup,
        'early_stop': args.early_stop,
        'seed': args.seed,
        'n_tested_imgs': args.n_tested_imgs,
        'victim_model': args.victim_model_name,
        'exp_dir': args.exp_dir,
    }

    exp_dir = args.exp_dir
    baseline = 'HillClimbing'
    if not args.step1_random and not args.step2_random:
        exp_dir = f'{exp_dir}/{baseline}_maxQuery-{args.max_query}_VictimModel-{args.victim_model_name}/Seed{args.seed}'
    else:
        config['step1_random'] = args.step1_random
        config['step2_random'] = args.step2_random

        exp_dir = (f'{exp_dir}/{baseline}_RandomStep1-{args.step1_random}_RandomStep2-{args.step2_random}_'
                   f'maxQuery-{args.max_query}_VictimModel-{args.victim_model_name}/Seed{args.seed}')

    os.makedirs(exp_dir, exist_ok=True)

    continue_exp = False
    if os.path.isfile(f'{exp_dir}/configs.json'):
        prev_config = json.load(open(f'{exp_dir}/configs.json'))
        continue_exp = checkSameConfigs(config, prev_config)

    if not continue_exp:
        with open(f'{exp_dir}/configs.json', "w") as file:
            json.dump(config, file, indent=4, cls=NumpyEncoder)

    exp_img_dir = f"{exp_dir}/images"
    exp_log_dir = f"{exp_dir}/logs"

    os.makedirs(exp_img_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)

    # Load pre-trained model for the face verification task
    victim_model_name = args.victim_model_name
    if victim_model_name in ['vggface', 'webface']:
        img_h, img_w = 160, 160
    else:  # arcface, cosface
        img_h, img_w = 112, 112
    MODEL = get_model(model_name=victim_model_name, model_dir=args.model_dir)
    print('Load Pre-trained model - Done!')

    # Load data
    n_tested_imgs = args.n_tested_imgs
    pair_path = './lfw_preprocess/pairs.txt'
    if victim_model_name != 'vggface':
        pair_path = f'./lfw_preprocess/{n_tested_imgs}pairs_{victim_model_name}.txt'
    DATA = LFW(IMG_DIR=args.img_dir, MASK_DIR=args.mask_dir, PAIR_PATH=pair_path, transform=None)
    print('Load Data - Done!')

    toTensor = transforms.ToTensor()

    random_seed = args.seed
    success_list = []
    for i in range(n_tested_imgs):
        print(f"Image #{i + 1}/{n_tested_imgs}")
        if os.path.isfile(f'{exp_log_dir}/{i}.p') and continue_exp:
            continue
        set_seed(random_seed)

        img1, img2, label = DATA[i]
        img1, img2 = img1.resize((img_h, img_w)), img2.resize((img_h, img_w))
        img1_torch, img2_torch = toTensor(img1), toTensor(img2)

        fitness = Fitness(img1=img1_torch, img2=img2_torch, model=MODEL, label=label,
                          recons_w=0.0, attack_w=0.0, fitness_type=None, multi_objective=False)

        # if not fitness.init_self_check():
        #     print(f'Image #{i + 1}/{n_tested_imgs} - Model gives wrong prediction at initialization!')
        #     continue

        best_psnr_success, best_ind_success = None, None

        algo = HillClimbing(max_query=args.max_query, img_h=img_h, img_w=img_w, patch_s=args.patch_size,
                            fitness=fitness, step1_random=args.step1_random, step2_random=args.step2_random,
                            n_warmup=args.n_warmup, early_stop=args.early_stop)

        best_patch = algo.solve()
        patch, loc = best_patch.patch, best_patch.location
        adv_img = fitness.apply_patch_to_image(patch, loc)

        adv_score, psnr_score = best_patch.adv_score.item(), best_patch.psnr_score.item()
        success_attack = (adv_score >= 0)
        success_list.append(success_attack)

        min_query = args.max_query
        if success_attack:
            adv_score_history = np.array([F[0] for F in algo.history])
            min_query = np.where(adv_score_history >= 0)[0][0] + 1

        print(f"Adv Score: {adv_score:.4f}")
        print(f"PSNR Score: {psnr_score:.4f}")
        print(f"Success Attack: {success_attack}")
        print(f"Min Query: {min_query}")
        print('-' * 20)

        # Save_image
        save_image(adv_img, f"{exp_img_dir}/{i}_Success-{success_attack}_AdvScore-{adv_score:.2f}_PSNRScore-{psnr_score:.2f}.png")

        # Save results
        results = {
            "loc": loc,
            "patch": patch.cpu().detach().numpy(),
            "cls_img": img1_torch.cpu().detach().numpy(),
            "ref_img": img2_torch.cpu().detach().numpy(),
            "adv_img": adv_img.cpu().detach().numpy(),
            "success_attack": success_attack,
            "adv_score": adv_score,
            "psnr_score": psnr_score,
            "min_query": min_query,
            "list_adv_psnr_scores": algo.history,
            "patch_before_refining": algo.patch_before_refining.cpu().detach().numpy(),
            "w": algo.w
        }
        p.dump(results, open(f'{exp_log_dir}/{i}.p', 'wb'))

    print(f"Success rate: {sum(success_list) / len(success_list)}")
