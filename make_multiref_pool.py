import os
import random
import argparse
import numpy as np
from tqdm import tqdm
 
from transfer_utils import (ALL_MODELS, THRESHOLD, get_device, load_model, load_img_tensor,
                            list_identity_ids, embed, cos_sim)
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 63)))
    p.add_argument('--n_pairs', type=int, default=100)
    p.add_argument('--min_refs', type=int, default=10, help='min number of references besides probe and A')
    p.add_argument('--n_refs', type=int, default=10, help='references kept per pair (0 = all valid ones)')
    p.add_argument('--max_probe_tries', type=int, default=5, help='probes tried per person before giving up')
    p.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    p.add_argument('--model_dir', type=str, default='./pretrained_model')
    p.add_argument('--check_models', type=str, nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument('--out_dir', type=str, default='./lfw_preprocess/multiref_pool')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()
 
 
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
 
    # ---------------- 1) pool of people ----------------
    people = sorted(d for d in os.listdir(args.img_dir) if os.path.isdir(os.path.join(args.img_dir, d)))
    ids = {n: list_identity_ids(args.img_dir, n) for n in people}
    pool = [n for n in people if len(ids[n]) >= args.min_refs + 2]
    pool_path = os.path.join(args.out_dir, f'people_min{args.min_refs}refs.txt')
    with open(pool_path, 'w') as f:
        f.write('name\tn_images\n' + ''.join(f'{n}\t{len(ids[n])}\n' for n in pool))
    print(f'{len(pool)} people have >= {args.min_refs + 2} images (>= {args.min_refs} besides probe and A) '
          f'-> {pool_path}')
 
    device = get_device(args.device)
    models = {m: load_model(m, args.model_dir, device) for m in args.check_models}
    cache = {}
 
    def feat(m, name, i):
        k = (m, name, i)
        if k not in cache:
            cache[k] = embed(models[m], load_img_tensor(args.img_dir, name, i, m), device)
        return cache[k]
 
    def ok(name, a, b):
        return all(cos_sim(feat(m, name, a), feat(m, name, b)).item() >= THRESHOLD for m in models)
 
    # ---------------- 2) random pairs per seed ----------------
    for seed in args.seeds:
        rng = random.Random(seed)
        order = pool[:]
        rng.shuffle(order)
        rows, used, rejected = [], [], 0
        queue = list(order)
        pbar = tqdm(total=args.n_pairs, desc=f'seed {seed}')
        while len(rows) < args.n_pairs and queue:
            name = queue.pop(0)
            done = False
            for _ in range(args.max_probe_tries):
                probe, a = rng.sample(ids[name], 2)
                if any(r[0] == name and r[1] == probe for r in rows) or not ok(name, probe, a):
                    continue
                refs = [c for c in ids[name] if c not in (probe, a) and ok(name, probe, c)]
                if len(refs) < args.min_refs:
                    continue
                if args.n_refs:
                    refs = sorted(rng.sample(refs, min(args.n_refs, len(refs))))
                rows.append((name, probe, a, refs))
                used.append(name)
                pbar.update(1)
                done = True
                break
            if not done:
                rejected += 1
            # pool exhausted before n_pairs: reuse people (different probe) in a new random order
            if not queue and len(rows) < args.n_pairs and used:
                print(f'\n[seed {seed}] WARNING: only {len(set(used))} people usable, reusing people')
                queue = list(dict.fromkeys(used))
                rng.shuffle(queue)
                if rejected > 50 * args.n_pairs:
                    break
        pbar.close()
 
        with open(os.path.join(args.out_dir, f'multiref_same_seed_{seed}.txt'), 'w') as f:
            f.write(f'multiref_same_seed_{seed}\n')
            f.writelines(f"{n}\t{p}\t{a}\t{','.join(map(str, r))}\n" for n, p, a, r in rows)
        with open(os.path.join(args.out_dir, f'same_seed_{seed}.txt'), 'w') as f:
            f.write(f'same_seed_{seed}\n')
            f.writelines(f'{n}\t{p}\t{a}\n' for n, p, a, _ in rows)
        n_refs = np.array([len(r[3]) for r in rows]) if rows else np.array([0])
        print(f'seed {seed}: {len(rows)} pairs from {len(set(used))} distinct people | '
              f'N refs: min {n_refs.min()}, max {n_refs.max()} | people rejected by clean check: {rejected}')
 
 
if __name__ == '__main__':
    main()