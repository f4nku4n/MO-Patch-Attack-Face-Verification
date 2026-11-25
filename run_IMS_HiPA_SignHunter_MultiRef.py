import os
import json
import argparse
import pickle as p

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from algorithm.IMS_HiPA_SignHunter_MultiRef import IMS_HiPA_SignHunter_MultiRef
from core.fitness_multiref_sh import MultiRefFitnessSH, AGGREGATIONS
from factory import get_model
from utils.common import set_seed, NumpyEncoder

MODEL_CFG = {  # threshold, img size (same as run_IMS_HiPA_SignHunter.py)
    'vggface': (0.5, 160), 'webface': (0.5, 160), 'cosface': (0.30, 112), 'arcface': (0.15, 112),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_query', type=int, default=10000, help="Maximum number of evaluations")
    parser.add_argument('--n_warmup', type=int, default=1)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--agg', type=str, default='lexi', choices=AGGREGATIONS,
                        help='how the N reference scores are folded into one adv_score')

    # IMS
    parser.add_argument('--b', type=int, default=2, help='base of IMS: attacker i makes 1 query every b^i iterations')
    parser.add_argument('--variant1', action='store_true', help='keep adding patch sizes after the first success')
    parser.add_argument('--variant2', action='store_true', help='after the first success, stop all other attackers')

    # SignHunter (same meaning as run_IMS_HiPA_SignHunter.py)
    parser.add_argument('--eps', type=int, default=32, help='step size (x/255)')
    parser.add_argument('--init', type=str, default='clean', choices=['clean', 'step1', 'gray'])
    parser.add_argument('--min_block', type=int, default=2)
    parser.add_argument('--no_split_channels', action='store_true')
    parser.add_argument('--no_reuse_sign', action='store_true')
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--eps_decay', type=float, default=0.5, help='<=0: off')
    parser.add_argument('--eps_min', type=int, default=1, help='minimum eps (x/255)')

    # Multi-reference data
    parser.add_argument('--list_dir', type=str, default='./lfw_preprocess',
                        help='folder with multiref_same_seed_<seed>.txt (e.g. .../multiref_pool/N5)')
    parser.add_argument('--max_refs', type=int, default=None, help='optional cap on N (default: all references)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='override the per-model threshold (0.5 = same as the old multiref runs)')
    parser.add_argument('--tag', type=str, default='', help='extra name for the run folder, e.g. N5')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--victim_model_name', type=str, default='vggface', choices=list(MODEL_CFG))
    parser.add_argument('--n_tested_imgs', type=int, default=100)
    parser.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    parser.add_argument('--model_dir', type=str, default='./pretrained_model')
    parser.add_argument('--exp_dir', type=str, default='./exp_results_multiref')
    return parser.parse_args()


def read_multiref(path):
    """Same format as main_multiref.py (branch Long): header, then  name  probe  A  ref1,ref2,..."""
    with open(path) as f:
        f.readline()
        rows = []
        for line in f:
            if not line.strip():
                continue
            name, probe, a, refs = (line.rstrip('\n').split('\t') + [''])[:4]
            rows.append((name, int(probe), int(a), [int(r) for r in refs.split(',') if r]))
    return rows


def load_img(img_dir, name, i, size, to_tensor):
    img = Image.open(os.path.join(img_dir, name, f'{name}_{str(i).zfill(4)}.jpg')).convert('RGB')
    return to_tensor(img.resize(size))


def final_stats(fitness, idv, ref_a_feat):
    """Scores of a final patch on the N references and on A (A: evaluation only, not counted as a query)."""
    with torch.no_grad():
        adv_img = fitness.apply_patch_to_image(idv.patch, idv.location)
        vec = fitness.adv_matrix(adv_img.unsqueeze(0))[0]
        feat = torch.nn.functional.normalize(fitness.model(adv_img.unsqueeze(0)), dim=1)
        adv_a = (fitness.threshold - (feat * ref_a_feat).sum()).item()
    return {'adv_vec': vec.cpu().numpy(), 'n_success': int((vec >= 0).sum()),
            'success_all': bool((vec >= 0).all()), 'frac_success': float((vec >= 0).float().mean()),
            'adv_A': adv_a, 'success_A': adv_a >= 0}, adv_img


def summarize(exp_dir, log_dir, n_pairs):
    logs = [p.load(open(f'{log_dir}/{i}.p', 'rb')) for i in range(n_pairs) if os.path.isfile(f'{log_dir}/{i}.p')]
    summ = {'n_pairs': len(logs)}
    if not logs:
        return summ
    for suf in ['', '_before_refining']:
        summ[f'ASR_all{suf}'] = float(np.mean([d['success_all' + suf] for d in logs]))
        summ[f'mean_n_success{suf}'] = float(np.mean([d['n_success' + suf] for d in logs]))
        summ[f'mean_frac_success{suf}'] = float(np.mean([d['frac_success' + suf] for d in logs]))
        summ[f'ASR_A{suf}'] = float(np.mean([d['success_A' + suf] for d in logs]))
        fooled_all = [d for d in logs if d['success_all' + suf]]
        summ[f'ASR_A_given_all{suf}'] = (float(np.mean([d['success_A' + suf] for d in fooled_all]))
                                        if fooled_all else float('nan'))
    succ = [d for d in logs if d['success_all']]
    summ['mean_n_refs'] = float(np.mean([d['n_refs'] for d in logs]))
    summ['mean_psnr'] = float(np.mean([d['psnr_score'] for d in logs]))
    summ['mean_psnr_success'] = float(np.mean([d['psnr_score'] for d in succ])) if succ else float('nan')
    summ['mean_min_query_success'] = float(np.mean([d['min_query'] for d in succ])) if succ else float('nan')
    summ['mean_patch_size_success'] = float(np.mean([d['best_patch_size'] for d in succ])) if succ else float('nan')
    json.dump(summ, open(f'{exp_dir}/summary.json', 'w'), indent=4)
    return summ


if __name__ == "__main__":
    args = parse_args()

    victim_model_name = args.victim_model_name
    threshold, img_s = MODEL_CFG[victim_model_name]
    if args.threshold is not None:
        threshold = args.threshold
    img_h = img_w = img_s

    MODEL = get_model(model_name=victim_model_name, model_dir=args.model_dir)
    print('Load Pre-trained model - Done!')

    config = vars(args).copy()
    config.update({'method': 'IMS_HiPA_SignHunter_MultiRef', 'threshold': threshold})

    tag = f'_{args.tag}' if args.tag else ''
    exp_dir = (f'{args.exp_dir}/IMS_HiPA_SignHunter_MultiRef{tag}_Agg-{args.agg}_maxQuery-{args.max_query}_'
               f'VictimModel-{victim_model_name}_Thr-{threshold}_b-{args.b}_Variant1-{args.variant1}_'
               f'Variant2-{args.variant2}_Eps-{args.eps}_Init-{args.init}_MinBlock-{args.min_block}_'
               f'EpsDecay-{args.eps_decay}/Seed{args.seed}')
    exp_img_dir, exp_log_dir = f"{exp_dir}/images", f"{exp_dir}/logs"
    os.makedirs(exp_img_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)
    with open(f'{exp_dir}/configs.json', "w") as file:
        json.dump(config, file, indent=4, cls=NumpyEncoder)

    toTensor = transforms.ToTensor()
    pairs = read_multiref(os.path.join(args.list_dir, f'multiref_same_seed_{args.seed}.txt'))
    n_pairs = min(args.n_tested_imgs, len(pairs))
    print(f'Loaded {n_pairs} pairs | agg = {args.agg} | threshold = {threshold}')

    for i in range(n_pairs):
        if os.path.isfile(f'{exp_log_dir}/{i}.p'):  # resume
            print(f'Pair #{i} - already done, skip')
            continue
        name, probe_id, a_id, ref_ids = pairs[i]
        if args.max_refs:
            ref_ids = ref_ids[:args.max_refs]
        if not ref_ids:
            print(f'Pair #{i}: no reference image -> skipped')
            continue
        set_seed(args.seed)

        probe = load_img(args.img_dir, name, probe_id, (img_h, img_w), toTensor)
        refs = [load_img(args.img_dir, name, r, (img_h, img_w), toTensor) for r in ref_ids]
        ref_a = load_img(args.img_dir, name, a_id, (img_h, img_w), toTensor)

        fitness = MultiRefFitnessSH(img1=probe, ref_imgs=refs, model=MODEL, threshold=threshold, agg=args.agg)
        with torch.no_grad():
            ref_a_feat = torch.nn.functional.normalize(MODEL(ref_a.unsqueeze(0).cuda()), dim=1)
        if not fitness.init_self_check():
            print(f'Pair #{i}: clean probe is not verified by every reference!')

        algo = IMS_HiPA_SignHunter_MultiRef(max_query=args.max_query, img_h=img_h, img_w=img_w, fitness=fitness,
                                            b=args.b, variant1=args.variant1, variant2=args.variant2,
                                            n_warmup=args.n_warmup, early_stop=args.early_stop,
                                            eps=args.eps / 255, init=args.init, min_block=args.min_block,
                                            split_channels=not args.no_split_channels,
                                            reuse_sign=not args.no_reuse_sign, shuffle=not args.no_shuffle,
                                            eps_decay=args.eps_decay if args.eps_decay > 0 else None,
                                            eps_min=args.eps_min / 255)
        best, best_patch_size, min_query = algo.solve()
        algo.pbar.close()

        final, adv_img = final_stats(fitness, best, ref_a_feat)
        before, _ = final_stats(fitness, algo.patch_before_refining, ref_a_feat)
        step2_queries = {ps: len(h) - 1 for ps, h in algo.history.items()}

        print(f'Pair #{i + 1}/{n_pairs} {name}: N={len(ref_ids)} | fooled {final["n_success"]}/{len(ref_ids)} '
              f'(before refining {before["n_success"]}) | all: {final["success_all"]} | '
              f'A: {"fooled" if final["success_A"] else "not fooled"} (adv {final["adv_A"]:.3f}) | '
              f'PSNR {best.psnr_score.item():.2f} | patch size {best_patch_size} | min query {min_query}')
        print(f"Step-2 queries per patch size: {step2_queries}")
        print('-' * 20)

        save_image(adv_img, f'{exp_img_dir}/{i}_Fooled-{final["n_success"]}of{len(ref_ids)}'
                            f'_A-{final["success_A"]}_PSNR-{best.psnr_score.item():.2f}_PatchS-{best_patch_size}.png')
        results = {
            'name': name, 'probe_id': probe_id, 'A_id': a_id, 'ref_ids': ref_ids, 'n_refs': len(ref_ids),
            'loc': best.location, 'patch': best.patch.cpu().numpy(),
            'patch_before_refining': algo.patch_before_refining.patch.cpu().numpy(),
            'cls_img': probe.numpy(), 'adv_img': adv_img.cpu().num py(),
            'adv_score': best.adv_score.item(), 'psnr_score': best.psnr_score.item(),
            'min_query': min_query, 'best_patch_size': best_patch_size,
            'w': algo.w, 'agg': args.agg, 'threshold': threshold,
            'list_adv_psnr_scores': algo.history, 'step2_queries': step2_queries,
            **final,
            **{f'{k}_before_refining': v for k, v in before.items()},
            'success_attack': final['success_all'],
        }
        p.dump(results, open(f'{exp_log_dir}/{i}.p', 'wb'))

    print(json.dumps(summarize(exp_dir, exp_log_dir, n_pairs), indent=4))