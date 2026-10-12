import os
import csv
import argparse
import numpy as np
from tqdm import tqdm

from transfer_utils import (ALL_MODELS, THRESHOLD, get_device, load_model, load_img_tensor,
                            list_identity_ids, read_pairs, embed, cos_sim)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 63)))
    p.add_argument('--transfer_dir', type=str, default='./lfw_preprocess',
                   help='folder with transfer_same_seed_<seed>_info.csv (100 pairs incl. replacements)')
    p.add_argument('--pair_dir', type=str, default=None,
                   help='use same_seed_<seed>.txt from this folder instead of the transfer info csv')
    p.add_argument('--out_dir', type=str, default='./lfw_preprocess')
    p.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    p.add_argument('--model_dir', type=str, default='./pretrained_model')
    p.add_argument('--check_models', type=str, nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def load_pairs(args, seed):
    if args.pair_dir:
        _, lines = read_pairs(os.path.join(args.pair_dir, f'same_seed_{seed}.txt'))
        return [(l[0], int(l[1]), int(l[2])) for l in lines]
    with open(os.path.join(args.transfer_dir, f'transfer_same_seed_{seed}_info.csv')) as f:
        rows = list(csv.DictReader(f))
    return [(r['name'], int(r['probe_id']), int(r['A_id'])) for r in rows if r['source'] != 'none']


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.device)
    models = {m: load_model(m, args.model_dir, device) for m in args.check_models}
    cache = {}

    def feat(m, name, i):
        k = (m, name, i)
        if k not in cache:
            cache[k] = embed(models[m], load_img_tensor(args.img_dir, name, i, m), device)
        return cache[k]

    for seed in args.seeds:
        pairs = load_pairs(args, seed)
        out, n_refs = [], []
        for name, probe, a in tqdm(pairs, desc=f'seed {seed}'):
            refs = []
            for c in list_identity_ids(args.img_dir, name):
                if c in (probe, a):
                    continue
                if all(cos_sim(feat(m, name, probe), feat(m, name, c)).item() >= THRESHOLD for m in models):
                    refs.append(c)
            n_refs.append(len(refs))
            out.append(f"{name}\t{probe}\t{a}\t{','.join(map(str, refs))}")
        path = os.path.join(args.out_dir, f'multiref_same_seed_{seed}.txt')
        with open(path, 'w') as f:
            f.write(f'multiref_same_seed_{seed}\n' + '\n'.join(out) + '\n')
        n = np.array(n_refs)
        print(f'seed {seed}: {len(n)} pairs | N refs per pair: min {n.min()}, median {int(np.median(n))}, '
              f'max {n.max()}, total {n.sum()} | pairs with 0 refs: {(n == 0).sum()} -> {path}')


if __name__ == '__main__':
    main()