import os
import csv
import argparse
import pickle as p
import numpy as np
import torch
from tqdm import tqdm

from transfer_utils import (ALL_MODELS, THRESHOLD, get_device, load_model, load_img_tensor,
                            read_pairs, embed, cos_sim)

PATCH_TYPES = ['patch', 'patch_before_refining']


def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('--victim_model_name', type=str, required=True, choices=ALL_MODELS)
    pa.add_argument('--optimizer', type=str, default='IMS-HiPA', choices=['HiPA', 'IMS-HiPA'])
    pa.add_argument('--variant1', action='store_true')
    pa.add_argument('--variant2', action='store_true')
    pa.add_argument('--max_query', type=int, default=10000)
    pa.add_argument('--exp_dir', type=str, default='./exp_results')
    pa.add_argument('--replace_exp_dir', type=str, default=None,
                    help='exp_dir used when running main_ours.py on same_seed_<seed>_replace.txt')
    pa.add_argument('--log_dir_template', type=str, default=None,
                    help='logs folder with {seed} placeholder; overrides --exp_dir/--optimizer naming')
    pa.add_argument('--replace_log_dir_template', type=str, default=None,
                    help='same, for the replacement pairs; overrides --replace_exp_dir')
    pa.add_argument('--method_name', type=str, default=None, help='name used for the output folder')
    pa.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 63)))
    pa.add_argument('--n_tested_imgs', type=int, default=100)
    pa.add_argument('--transfer_dir', type=str, default='./lfw_preprocess',
                    help='folder with transfer_same_seed_<seed>_info.csv')
    pa.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    pa.add_argument('--model_dir', type=str, default='./pretrained_model')
    pa.add_argument('--out_dir', type=str, default='./transfer_results')
    pa.add_argument('--device', type=str, default=None)
    return pa.parse_args()


def default_log_dir(args, exp_dir):
    """Reproduces the folder name built in main_ours.py (setting = same)."""
    baseline = args.optimizer
    if args.optimizer == 'IMS-HiPA':
        baseline += f'_Variant1-{args.variant1}_Variant2-{args.variant2}'
    return (f'{exp_dir}/{baseline}_Variant1-{args.variant1}_Variant2-{args.variant2}_'
            f'maxQuery-{args.max_query}_Settingsame_VictimModel-{args.victim_model_name}/Seed{{seed}}/logs')


def paste(cls_img, patch, loc):
    img = cls_img.clone()
    x0, x1, y0, y1 = [int(v) for v in loc]
    img[:, x0:x1, y0:y1] = torch.as_tensor(np.asarray(patch), dtype=img.dtype)
    return img


def mean_std(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    if not xs:
        return float('nan'), float('nan')
    return float(np.mean(xs)), float(np.std(xs))


def main():
    args = parse_args()
    m = args.victim_model_name
    method = args.method_name or (args.optimizer if args.optimizer == 'HiPA'
                                  else f'IMS-HiPA_V1-{args.variant1}_V2-{args.variant2}')
    template = args.log_dir_template or default_log_dir(args, args.exp_dir)
    rep_template = args.replace_log_dir_template or (
        default_log_dir(args, args.replace_exp_dir) if args.replace_exp_dir else None)
    out_dir = os.path.join(args.out_dir, f'{method}_{m}')
    os.makedirs(out_dir, exist_ok=True)

    device = get_device(args.device)
    model = load_model(m, args.model_dir, device)
    print(f'Device: {device} | model: {m}\n  orig logs:    {template}\n  replace logs: {rep_template}')

    pair_rows, seed_rows = [], []
    warn = {'missing_pickle': 0, 'missing_before_refining': 0, 'pickle_vs_dataset_mismatch': 0,
            'adv_score_mismatch': 0, 'no_new_ref': 0, 'replace_dir_not_given': 0}

    for seed in args.seeds:
        with open(os.path.join(args.transfer_dir, f'transfer_same_seed_{seed}_info.csv')) as f:
            info = list(csv.DictReader(f))

        seed_pairs = []
        for r in tqdm(info[:args.n_tested_imgs], desc=f'seed {seed}'):
            i, source = int(r['idx']), r['source']
            name, probe_id, ref_a, ref_b = r['name'], r['probe_id'], r['A_id'], r['B_id']
            if source == 'none' or ref_b == 'NA':
                warn['no_new_ref'] += 1
                continue
            if source == 'replace':
                if rep_template is None:
                    warn['replace_dir_not_given'] += 1
                    continue
                log_dir = rep_template.format(seed=seed)
            else:
                log_dir = template.format(seed=seed)
            pk = os.path.join(log_dir, f"{r['log_idx']}.p")
            if not os.path.isfile(pk):
                warn['missing_pickle'] += 1
                continue
            d = p.load(open(pk, 'rb'))

            cls_img = torch.as_tensor(np.asarray(d['cls_img']), dtype=torch.float32)
            ref_a_img = torch.as_tensor(np.asarray(d['ref_img']), dtype=torch.float32)

            # sanity: pickle i really is pair i of same_seed_<seed>.txt
            ds_probe = load_img_tensor(args.img_dir, name, probe_id, m)
            ds_ref_a = load_img_tensor(args.img_dir, name, ref_a, m)
            if (cls_img - ds_probe).abs().max() > 1e-4 or (ref_a_img - ds_ref_a).abs().max() > 1e-4:
                warn['pickle_vs_dataset_mismatch'] += 1

            ref_b_img = load_img_tensor(args.img_dir, name, ref_b, m)
            f_a, f_b = embed(model, ref_a_img, device), embed(model, ref_b_img, device)
            f_clean = embed(model, cls_img, device)
            clean_a, clean_b = cos_sim(f_clean, f_a).item(), cos_sim(f_clean, f_b).item()

            for pt in PATCH_TYPES:
                if pt not in d or d[pt] is None:
                    if pt == 'patch_before_refining':
                        warn['missing_before_refining'] += 1
                    continue
                adv = paste(cls_img, d[pt], d['loc'])
                f_adv = embed(model, adv, device)
                sim_a, sim_b = cos_sim(f_adv, f_a).item(), cos_sim(f_adv, f_b).item()
                if pt == 'patch' and 'adv_score' in d and abs((THRESHOLD - sim_a) - float(d['adv_score'])) > 1e-3:
                    warn['adv_score_mismatch'] += 1
                row = {'seed': seed, 'idx': i, 'source': source, 'name': name, 'probe_id': probe_id, 'ref_A': ref_a, 'ref_B': ref_b,
                       'patch_type': pt, 'patch_size': int(d['loc'][1]) - int(d['loc'][0]),
                       'clean_sim_A': clean_a, 'clean_sim_B': clean_b, 'sim_A': sim_a, 'sim_B': sim_b,
                       'success_A': int(sim_a < THRESHOLD), 'success_B': int(sim_b < THRESHOLD),
                       'clean_B_correct': int(clean_b >= THRESHOLD),
                       'stored_success_A': int(bool(d.get('success_attack', False)))}
                seed_pairs.append(row)
        pair_rows += seed_pairs

        for pt in PATCH_TYPES:
            rows = [r for r in seed_pairs if r['patch_type'] == pt]
            if not rows:
                continue
            sa = [r for r in rows if r['success_A']]
            seed_rows.append({
                'seed': seed, 'patch_type': pt, 'n_pairs': len(rows),
                'ASR_A': np.mean([r['success_A'] for r in rows]),
                'ASR_B': np.mean([r['success_B'] for r in rows]),
                'transfer_rate': np.mean([r['success_B'] for r in sa]) if sa else float('nan'),
                'mean_sim_A': np.mean([r['sim_A'] for r in rows]),
                'mean_sim_B': np.mean([r['sim_B'] for r in rows]),
                'mean_clean_sim_B': np.mean([r['clean_sim_B'] for r in rows]),
            })

    def dump(path, rows):
        if rows:
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    dump(os.path.join(out_dir, 'per_pair.csv'), pair_rows)
    dump(os.path.join(out_dir, 'per_seed.csv'), seed_rows)

    summary = []
    metrics = ['ASR_A', 'ASR_B', 'transfer_rate', 'mean_sim_A', 'mean_sim_B']
    for pt in PATCH_TYPES:
        rows = [r for r in seed_rows if r['patch_type'] == pt]
        if not rows:
            continue
        s = {'method': method, 'victim_model': m, 'patch_type': pt, 'n_seeds': len(rows),
             'n_pairs_total': sum(r['n_pairs'] for r in rows)}
        for k in metrics:
            mu, sd = mean_std([r[k] for r in rows])
            s[f'{k}_mean'], s[f'{k}_std'] = round(mu, 4), round(sd, 4)
        summary.append(s)
    dump(os.path.join(out_dir, 'summary.csv'), summary)

    print(f'\n=== {method} | {m} | mean (std) over seeds ===')
    for s in summary:
        print(f"{s['patch_type']:<22} ASR_A {s['ASR_A_mean']:.4f} ({s['ASR_A_std']:.4f})  "
              f"ASR_B {s['ASR_B_mean']:.4f} ({s['ASR_B_std']:.4f})  "
              f"transfer {s['transfer_rate_mean']:.4f}  n={s['n_pairs_total']}")
    print('\nChecks:', ', '.join(f'{k}={v}' for k, v in warn.items()))
    if warn['pickle_vs_dataset_mismatch'] or warn['adv_score_mismatch']:
        print('WARNING: some pickles do not match the pair list / stored scores - check --log_dir_template, '
              '--img_dir and that results were produced on same_seed_<seed>.txt.')
    print(f'Saved to {out_dir}')


if __name__ == '__main__':
    main()