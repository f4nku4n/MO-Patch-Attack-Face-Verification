import os
import json
import argparse
import pickle as p

from core import Fitness
from algorithm.IMS_HiPA_SignHunter_Selected import IMS_HiPA_SignHunter_Selected

from dataset import LFW
from factory import get_model

from torchvision import transforms
from torchvision.utils import save_image

from utils.common import set_seed, NumpyEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_query', type=int, default=10000, help="Maximum number of evaluations")
    parser.add_argument('--n_warmup', type=int, default=1)
    parser.add_argument('--early_stop', action='store_true', help='Stop as soon as the attack succeeds (no Step-3)')

    # IMS
    parser.add_argument('--b', type=int, default=2, help='base of IMS: attacker i makes 1 query every b^i iterations')
    parser.add_argument('--variant1', action='store_true', help='keep adding patch sizes after the first success')
    parser.add_argument('--variant2', action='store_true', help='after the first success, stop all other attackers')

    # SignHunter (same meaning as run_HiPA_SignHunter_Test.py)
    parser.add_argument('--eps', type=int, default=32, help='step size (x/255)')
    parser.add_argument('--init', type=str, default='clean', choices=['clean', 'step1', 'gray'])
    parser.add_argument('--min_block', type=int, default=2)
    parser.add_argument('--no_split_channels', action='store_true')
    parser.add_argument('--no_reuse_sign', action='store_true')
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--eps_decay', type=float, default=0.5, help='<=0: off')
    parser.add_argument('--eps_min', type=int, default=1, help='minimum eps (x/255)')

    # Images
    parser.add_argument('--img_ids', type=int, nargs='+', default=None, help='only attack these image indices')
    parser.add_argument('--ifgsm_log_dir', type=str, default=None,
                        help='<iFGSM exp_dir>/logs -> only attack images where iFGSM succeeded (location is NOT reused)')

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


def select_img_ids(args):
    img_ids = args.img_ids if args.img_ids is not None else list(range(args.n_tested_imgs))
    if args.ifgsm_log_dir is not None:
        img_ids = [i for i in img_ids if os.path.isfile(f'{args.ifgsm_log_dir}/{i}.p')
                   and p.load(open(f'{args.ifgsm_log_dir}/{i}.p', 'rb'))['success_attack']]
    return img_ids


if __name__ == "__main__":
    args = parse_args()

    victim_model_name = args.victim_model_name
    MODEL = get_model(model_name=victim_model_name, model_dir=args.model_dir)
    print('Load Pre-trained model - Done!')
    if victim_model_name in ['webface', 'vggface']:
        threshold, img_h, img_w = 0.5, 160, 160
    elif victim_model_name == 'cosface':
        threshold, img_h, img_w = 0.30, 112, 112
    elif victim_model_name == 'arcface':
        threshold, img_h, img_w = 0.15, 112, 112
    else:
        raise ValueError

    config = vars(args).copy()
    config.update({'method': 'IMS_HiPA_SignHunter', 'threshold': threshold})

    exp_dir = (f'{args.exp_dir}/IMS_HiPA_SignHunter_maxQuery-{args.max_query}_Setting-{args.setting}_'
               f'VictimModel-{victim_model_name}_b-{args.b}_Variant1-{args.variant1}_Variant2-{args.variant2}_Eps-{args.eps}_Init-{args.init}_'
               f'MinBlock-{args.min_block}_EpsDecay-{args.eps_decay}/Seed{args.seed}')
    exp_img_dir, exp_log_dir = f"{exp_dir}/images", f"{exp_dir}/logs"
    os.makedirs(exp_img_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)
    with open(f'{exp_dir}/configs.json', "w") as file:
        json.dump(config, file, indent=4, cls=NumpyEncoder)

    pair_path = f'./lfw_preprocess/{args.setting}_seed_{args.seed}.txt'
    DATA = LFW(IMG_DIR=args.img_dir, MASK_DIR=args.mask_dir, PAIR_PATH=pair_path, transform=None)
    print('Load Data - Done!')
    toTensor = transforms.ToTensor()

    img_ids = select_img_ids(args)
    print(f'Attack {len(img_ids)} images: {img_ids}')
    for i in img_ids:
        if os.path.isfile(f'{exp_log_dir}/{i}.p'):  # resume
            print(f'Image #{i} - already done, skip')
            continue
        print(f"Image #{i}")
        set_seed(args.seed)

        img1, img2, label = DATA[i]
        img1, img2 = img1.resize((img_h, img_w)), img2.resize((img_h, img_w))
        img1_torch, img2_torch = toTensor(img1), toTensor(img2)

        fitness = Fitness(img1=img1_torch, img2=img2_torch, model=MODEL, label=label,
                          recons_w=0.0, attack_w=0.0, fitness_type=None, threshold=threshold)
        if not fitness.init_self_check():
            print(f'Image #{i} - Model gives wrong prediction at initialization!')

        algo = IMS_HiPA_SignHunter_Selected(img_ref = img2, max_query=args.max_query, img_h=img_h, img_w=img_w, fitness=fitness,
                                   b=args.b, variant1=args.variant1, variant2=args.variant2,
                                   n_warmup=args.n_warmup, early_stop=args.early_stop,
                                   eps=args.eps / 255, init=args.init, min_block=args.min_block,
                                   split_channels=not args.no_split_channels, reuse_sign=not args.no_reuse_sign,
                                   shuffle=not args.no_shuffle,
                                   eps_decay=args.eps_decay if args.eps_decay > 0 else None,
                                   eps_min=args.eps_min / 255)
        best_patch, best_patch_size, min_query = algo.solve()
        algo.pbar.close()

        patch, loc = best_patch.patch, best_patch.location
        adv_img = fitness.apply_patch_to_image(patch, loc)
        adv_score, psnr_score = best_patch.adv_score.item(), best_patch.psnr_score.item()
        success_attack = (adv_score >= 0)
        # history[p] = 1 score after Step-1 + 1 score per Step-2 query of the HiPA with patch size p
        step2_queries = {ps: len(h) - 1 for ps, h in algo.history.items()}

        print(f"Adv Score: {adv_score:.4f} | PSNR: {psnr_score:.4f} | Success: {success_attack} | "
              f"Min Query: {min_query} | Patch size: {best_patch_size} | Loc: {loc}")
        print(f"Step-2 queries per patch size: {step2_queries}")
        print('-' * 20)

        save_image(adv_img, f"{exp_img_dir}/{i}_Success-{success_attack}_AdvScore-{adv_score:.2f}_"
                            f"PSNRScore-{psnr_score:.2f}_PatchS-{best_patch_size}.png")
        results = {
            "loc": loc,
            "patch": patch.cpu().detach().numpy(),
            "adv_img": adv_img.cpu().detach().numpy(),
            "success_attack": success_attack,
            "adv_score": adv_score,
            "psnr_score": psnr_score,
            "min_query": min_query,
            "best_patch_size": best_patch_size,
            "list_adv_psnr_scores": algo.history,
            "patch_before_refining": algo.patch_before_refining.patch.cpu().detach().numpy(),
            "w": algo.w,
            "step2_queries": step2_queries,
        }
        p.dump(results, open(f'{exp_log_dir}/{i}.p', 'wb'))

    done = [p.load(open(f'{exp_log_dir}/{i}.p', 'rb')) for i in img_ids if os.path.isfile(f'{exp_log_dir}/{i}.p')]
    n_success = sum(r['success_attack'] for r in done)
    print(f"Success rate: {n_success}/{len(done)} = {n_success / max(len(done), 1):.4f}")
