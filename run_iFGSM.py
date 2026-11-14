import os
import json
import argparse
import pickle as p

from core import Fitness

from dataset import LFW
from factory import get_model

from torchvision import transforms
from torchvision.utils import save_image

from utils.common import set_seed, NumpyEncoder, checkSameConfigs

from algorithm.FGSM import iFGSM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_query', type=int, default=1000)
    parser.add_argument('--eps', type=int, default=32)
    parser.add_argument('--alpha', type=int, default=32)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--victim_model_name', type=str, default='vggface',
                        choices=['vggface', 'webface', 'arcface', 'cosface'],
                        help='pretrained victim model')
    parser.add_argument('--n_tested_imgs', type=int, default=100, help="the number of tested images")

    parser.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    parser.add_argument('--model_dir', type=str, default='./pretrained_model')
    parser.add_argument('--mask_dir', type=str, default='./mask')
    parser.add_argument('--exp_dir', type=str, default='./exp_results')
    parser.add_argument('--setting', type=str, default='same', choices=['same', 'different'])
    parser.add_argument('--full_image', action='store_true', help="whether to use the full image for attack")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    max_query = args.max_query

    config = {
        'method': 'iFGSM',
        'max_query': args.max_query,
        'alpha': f'{args.alpha}/255',
        'eps': f'{args.eps}/255',
        'seed': args.seed,
        'n_tested_imgs': args.n_tested_imgs,
        'victim_model': args.victim_model_name,
        'exp_dir': args.exp_dir,
        'setting': args.setting,
        'full_image': args.full_image,
    }

    # Load pre-trained model for the face verification task
    victim_model_name = args.victim_model_name
    MODEL = get_model(model_name=victim_model_name, model_dir=args.model_dir)
    print('Load Pre-trained model - Done!')
    if victim_model_name in ['webface', 'vggface']:
        threshold = 0.5
        img_h, img_w = 160, 160
        patch_s = 20
    elif victim_model_name == 'cosface':
        threshold = 0.30
        img_h, img_w = 112, 112
        patch_s = 24
    elif victim_model_name == 'arcface':
        threshold = 0.15
        img_h, img_w = 112, 112
        patch_s = 24
    else:
        raise ValueError

    exp_dir = (f'{args.exp_dir}/iFGSM_'
               f'maxQuery-{args.max_query}_Setting-{args.setting}_VictimModel-{args.victim_model_name}_'
               f'Alpha-{args.alpha}_FullImage-{args.full_image}/Seed{args.seed}')

    os.makedirs(exp_dir, exist_ok=True)

    continue_exp = False
    if os.path.isfile(f'{exp_dir}/configs.json'):
        prev_config = json.load(open(f'{exp_dir}/configs.json'))
        continue_exp = checkSameConfigs(config, prev_config)

    print(config)
    if not continue_exp:
        with open(f'{exp_dir}/configs.json', "w") as file:
            json.dump(config, file, indent=4, cls=NumpyEncoder)

    exp_img_dir = f"{exp_dir}/images"
    exp_log_dir = f"{exp_dir}/logs"

    os.makedirs(exp_img_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)

    # Load data
    n_tested_imgs = args.n_tested_imgs
    pair_path = f'./lfw_preprocess/{args.setting}_seed_{args.seed}.txt'
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
                          recons_w=0.0, attack_w=0.0, fitness_type=None, threshold=threshold)

        if not fitness.init_self_check():
            print(f'Image #{i + 1}/{n_tested_imgs} - Model gives wrong prediction at initialization!')

        # algo = iFGSM(max_query=max_query, img_h=img_h, img_w=img_w, patch_s=patch_s, fitness=fitness)
        loc = None
        if args.full_image:
            loc = (0, img_h, 0, img_w)
        algo = iFGSM(
            max_query=max_query, img_h=img_h, img_w=img_w, patch_s=patch_s,
            fitness=fitness, eps=args.eps / 255, alpha=args.alpha / 255,
            predefined_loc=loc
        )

        best_patch, best_patch_size, min_query = algo.solve()
        patch, loc = best_patch.patch, best_patch.location
        adv_img = fitness.apply_patch_to_image(patch, loc)

        adv_score, psnr_score = best_patch.adv_score.item(), best_patch.psnr_score.item()
        success_attack = (adv_score >= 0)
        success_list.append(success_attack)

        print(f"Adv Score: {adv_score:.4f}")
        print(f"PSNR Score: {psnr_score:.4f}")
        print(f"Success Attack: {success_attack}")
        print(f"Min Query: {min_query}")
        print(f"The best patch size: {best_patch_size}")
        print('-' * 20)

        # Save_image
        save_image(adv_img,
                   f"{exp_img_dir}/{i}_Success-{success_attack}_AdvScore-{adv_score:.2f}_PSNRScore-{psnr_score:.2f}.png")

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
        }
        p.dump(results, open(f'{exp_log_dir}/{i}.p', 'wb'))

    print(f"Success rate: {sum(success_list) / len(success_list)}")