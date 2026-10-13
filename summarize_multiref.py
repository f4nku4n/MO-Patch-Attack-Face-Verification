import os
import glob
import json
import argparse
import numpy as np
 
KEYS = ['ASR_all', 'mean_n_success', 'mean_frac_success', 'ASR_all_before_refining',
        'mean_n_success_before_refining', 'mean_frac_success_before_refining', 'ASR_A', 'ASR_A_given_all',
        'ASR_A_before_refining', 'ASR_A_given_all_before_refining', 'mean_psnr', 'mean_n_refs']
 
if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--exp_dir', type=str, required=True)
    args = pa.parse_args()
    rows = []
    for run in sorted(glob.glob(os.path.join(args.exp_dir, 'IMS-HiPA-MultiRef*'))):
        seeds = [json.load(open(f)) for f in sorted(glob.glob(f'{run}/Seed*/summary.json'))]
        if not seeds:
            continue
        row = {'run': os.path.basename(run), 'n_seeds': len(seeds)}
        for k in KEYS:
            vals = [s[k] for s in seeds if k in s and s[k] == s[k]]
            row[k] = f'{np.mean(vals):.4f} ({np.std(vals):.4f})' if vals else ''
        rows.append(row)
        print(row['run'], f"| seeds {row['n_seeds']}")
        for k in KEYS:
            print(f'   {k:<36} {row[k]}')
    if rows:
        import csv
        out = os.path.join(args.exp_dir, 'all_summary.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print('Saved', out)