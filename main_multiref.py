import os
import json
import argparse
import pickle as p
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from algorithm import IMS_HiPA
from core.fitness_multiref import MultiRefFitness, AGGREGATIONS
from factory import get_model
from utils.common import set_seed, NumpyEncoder


def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('--max_query', type=int, default=10000)
    pa.add_argument('--agg', type=str, default='max', choices=AGGREGATIONS,
                    help='how to compare candidates over N references')
    pa.add_argument('--n_warmup', type=int, default=1)
    pa.add_argument('--b', type=int, default=2, help='base value of IMS-HiPA')
    pa.add_argument('--variant1', action='store_true')
    pa.add_argument('--variant2', action='store_true')
    pa.add_argument('--early_stop', action='store_true')
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--victim_model_name', type=str, default='vggface',
                    choices=['vggface', 'webface', 'arcface', 'cosface'])
    pa.add_argument('--n_tested_imgs', type=int, default=100)
    pa.add_argument('--max_refs', type=int, default=None, help='optional cap on N (default: all references)')
    pa.add_argument('--list_dir', type=str, default='./lfw_preprocess',
                    help='folder with multiref_same_seed_<seed>.txt')
    pa.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    pa.add_argument('--model_dir', type=str, default='./pretrained_model')
    pa.add_argument('--exp_dir', type=str, default='./exp_results_multiref')
    pa.add_argument('--tag', type=str, default='', help='extra name for the run folder, e.g. N5')
    return pa.parse_args()


def read_multiref(path):
    with open(path) as f:
        f.readline()
        rows = []
        for line in f:
            if not line.strip():
                continue
            name, probe, a, refs = (line.rstrip('\n').split('\t') + [''])[:4]
            rows.append((name, int(probe), int(a), [int(r) for r in refs.split(',') if r]))
    return rows


def load(img_dir, name, i, size, to_tensor):
    img = Image.open(os.path.join(img_dir, name, f'{name}_{str(i).zfill(4)}.jpg')).convert('RGB')
    return to_tensor(img.resize(size))


def stats(fitness, idv, ref_a_feat):
    """Scores of a final individual on the N references (+ A, evaluation only, not counted as a query)."""
    with torch.no_grad():
        adv_img = fitness.apply_patch_to_image(idv.patch, idv.location)
        vec = fitness.adv_matrix(adv_img.unsqueeze(0))[0]
        f = torch.nn.functional.normalize(fitness.model(adv_img.unsqueeze(0)), dim=1)
        adv_a = (0.5 - (f * ref_a_feat).sum()).item()
    return {'adv_vec': vec.cpu().numpy(), 'n_success': int((vec >= 0).sum()),
            'success_all': bool((vec >= 0).all()), 'frac_success': float((vec >= 0).float().mean()),
            'adv_A': adv_a, 'success_A': adv_a >= 0}, adv_img


def main():
    args = parse_args()
    v = f'_Variant1-{args.variant1}_Variant2-{args.variant2}'
    tag = f'_{args.tag}' if args.tag else ''
    exp_dir = (f'{args.exp_dir}/IMS-HiPA-MultiRef{tag}{v}_Agg-{args.agg}_maxQuery-{args.max_query}'
               f'_VictimModel-{args.victim_model_name}/Seed{args.seed}')
    img_dir_out, log_dir = f'{exp_dir}/images', f'{exp_dir}/logs'
    os.makedirs(img_dir_out, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    json.dump(vars(args), open(f'{exp_dir}/configs.json', 'w'), indent=4, cls=NumpyEncoder)

    m = args.victim_model_name
    size = (160, 160) if m in ['vggface', 'webface'] else (112, 112)
    model = get_model(model_name=m, model_dir=args.model_dir)
    to_tensor = transforms.ToTensor()

    pairs = read_multiref(os.path.join(args.list_dir, f'multiref_same_seed_{args.seed}.txt'))
    n_pairs = min(args.n_tested_imgs, len(pairs))
    print(f'Loaded {n_pairs} pairs | agg = {args.agg}')

    for i in range(n_pairs):
        if os.path.isfile(f'{log_dir}/{i}.p'):
            continue
        name, probe_id, a_id, ref_ids = pairs[i]
        if args.max_refs:
            ref_ids = ref_ids[:args.max_refs]
        if not ref_ids:
            print(f'Pair #{i}: no reference image -> skipped')
            continue
        set_seed(args.seed)

        probe = load(args.img_dir, name, probe_id, size, to_tensor)
        refs = [load(args.img_dir, name, r, size, to_tensor) for r in ref_ids]
        ref_a = load(args.img_dir, name, a_id, size, to_tensor)

        fitness = MultiRefFitness(img1=probe, ref_imgs=refs, model=model, agg=args.agg)
        with torch.no_grad():
            ref_a_feat = torch.nn.functional.normalize(model(ref_a.unsqueeze(0).cuda()), dim=1)
        if not fitness.init_self_check():
            print(f'Pair #{i}: clean probe is not verified by every reference!')

        algo = IMS_HiPA(max_query=args.max_query, img_h=size[0], img_w=size[1], fitness=fitness, b=args.b,
                        variant1=args.variant1, variant2=args.variant2, n_warmup=args.n_warmup,
                        early_stop=args.early_stop)
        best, best_patch_size, min_query = algo.solve()
        algo.pbar.close()

        final, adv_img = stats(fitness, best, ref_a_feat)
        before, _ = stats(fitness, algo.patch_before_refining, ref_a_feat)

        print(f'Pair #{i + 1}/{n_pairs} {name}: N={len(ref_ids)} | fooled {final["n_success"]}/{len(ref_ids)} '
              f'(before refining {before["n_success"]}) | all: {final["success_all"]} | '
              f'A: {"fooled" if final["success_A"] else "not fooled"} (adv {final["adv_A"]:.3f}, '
              f'before refining {"fooled" if before["success_A"] else "not fooled"}) | '
              f'PSNR {best.psnr_score.item():.4f} | patch size {best_patch_size} | min query {min_query}')

        save_image(adv_img, f'{img_dir_out}/{i}_Fooled-{final["n_success"]}of{len(ref_ids)}.png')
        results = {
            'name': name, 'probe_id': probe_id, 'A_id': a_id, 'ref_ids': ref_ids, 'n_refs': len(ref_ids),
            'loc': best.location, 'patch': best.patch.cpu().numpy(),
            'patch_before_refining': algo.patch_before_refining.patch.cpu().numpy(),
            'cls_img': probe.numpy(), 'adv_img': adv_img.cpu().numpy(),
            'psnr_score': best.psnr_score.item(), 'min_query': min_query, 'best_patch_size': best_patch_size,
            'w': algo.w, 'agg': args.agg, 'list_adv_psnr_scores': algo.history,
            **{k: v for k, v in final.items()},
            **{f'{k}_before_refining': v for k, v in before.items()},
            'success_attack': final['success_all'],
        }
        p.dump(results, open(f'{log_dir}/{i}.p', 'wb'))

    # per-seed summary
    logs = [p.load(open(f'{log_dir}/{i}.p', 'rb')) for i in range(n_pairs) if os.path.isfile(f'{log_dir}/{i}.p')]
    summ = {'n_pairs': len(logs)}
    for suf in ['', '_before_refining']:
        summ[f'ASR_all{suf}'] = float(np.mean([d['success_all' + suf] for d in logs]))
        summ[f'mean_n_success{suf}'] = float(np.mean([d['n_success' + suf] for d in logs]))
        summ[f'mean_frac_success{suf}'] = float(np.mean([d['frac_success' + suf] for d in logs]))
        summ[f'ASR_A{suf}'] = float(np.mean([d['success_A' + suf] for d in logs]))
        fooled_all = [d for d in logs if d['success_all' + suf]]
        summ[f'ASR_A_given_all{suf}'] = (float(np.mean([d['success_A' + suf] for d in fooled_all]))
                                        if fooled_all else float('nan'))
    summ['mean_n_refs'] = float(np.mean([d['n_refs'] for d in logs]))
    summ['mean_psnr'] = float(np.mean([d['psnr_score'] for d in logs]))
    json.dump(summ, open(f'{exp_dir}/summary.json', 'w'), indent=4)
    print(json.dumps(summ, indent=4))


if __name__ == '__main__':
    main()