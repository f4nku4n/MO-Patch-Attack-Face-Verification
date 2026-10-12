import os
import csv
import random
import argparse
from tqdm import tqdm

from transfer_utils import (ALL_MODELS, THRESHOLD, get_device, load_model, load_img_tensor,
                            list_identity_ids, read_pairs, embed, cos_sim)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 63)))
    p.add_argument('--pair_dir', type=str, default='./lfw_preprocess', help='folder with same_seed_<seed>.txt')
    p.add_argument('--out_dir', type=str, default='./lfw_preprocess')
    p.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    p.add_argument('--model_dir', type=str, default='./pretrained_model')
    p.add_argument('--check_models', type=str, nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument('--common_same', type=str, default=None,
                   help='optional common_same.txt; if given, replacement pairs are drawn from it')
    p.add_argument('--no_replace', action='store_true', help='write NA instead of replacing (old behaviour)')
    p.add_argument('--max_triplets_per_person', type=int, default=10)
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.device)
    models = {m: load_model(m, args.model_dir, device) for m in args.check_models}
    print(f'Device: {device} | models used for the clean check: {list(models)}')

    cache = {}

    def feat(m, name, img_id):
        key = (m, name, int(img_id))
        if key not in cache:
            cache[key] = embed(models[m], load_img_tensor(args.img_dir, name, img_id, m), device)
        return cache[key]

    def clean_sims(name, a, b):
        return {m: cos_sim(feat(m, name, a), feat(m, name, b)).item() for m in models}

    def all_ok(s):
        return all(v >= THRESHOLD for v in s.values())

    # People that can provide a replacement: >= 3 images
    all_people = sorted(d for d in os.listdir(args.img_dir) if os.path.isdir(os.path.join(args.img_dir, d)))
    ids_of = {}

    def ids(name):
        if name not in ids_of:
            ids_of[name] = list_identity_ids(args.img_dir, name)
        return ids_of[name]

    common = None
    if args.common_same:
        _, common = read_pairs(args.common_same)
        common = [l for l in common if len(l) == 3]

    def find_B(name, probe, a, rng_key):
        cands = [c for c in ids(name) if c not in (probe, a)]
        random.Random(rng_key).shuffle(cands)
        for c in cands:
            s = clean_sims(name, probe, c)
            if all_ok(s):
                return c, s, len(cands)
        return None, {}, len(cands)

    for seed in args.seeds:
        _, lines = read_pairs(os.path.join(args.pair_dir, f'same_seed_{seed}.txt'))
        used_people = {l[0] for l in lines}

        # replacement pool for this seed (deterministic order)
        if common is not None:
            pool = [(l[0], int(l[1]), int(l[2])) for l in common if l[0] not in used_people]
            random.Random(f'pool-{seed}').shuffle(pool)
        else:
            pool = [n for n in all_people if n not in used_people]
            random.Random(f'pool-{seed}').shuffle(pool)
        pool_pos = 0

        def next_replacement():
            nonlocal pool_pos
            while pool_pos < len(pool):
                item = pool[pool_pos]
                pool_pos += 1
                if common is not None:
                    name, probe, a = item
                    if len(ids(name)) < 3 or name in used_people:
                        continue
                    b, s_b, _ = find_B(name, probe, a, f'{seed}-rep-{name}')
                    if b is not None:
                        return name, probe, a, b, clean_sims(name, probe, a), s_b
                else:
                    name = item
                    pid = ids(name)
                    if len(pid) < 3:
                        continue
                    rng = random.Random(f'{seed}-rep-{name}')
                    for _ in range(args.max_triplets_per_person):
                        probe, a = rng.sample(pid, 2)
                        s_a = clean_sims(name, probe, a)
                        if not all_ok(s_a):
                            continue
                        b, s_b, _ = find_B(name, probe, a, f'{seed}-rep-{name}-{probe}-{a}')
                        if b is not None:
                            return name, probe, a, b, s_a, s_b
            return None

        out_lines, info, rep_lines = [], [], []
        stats = {'orig': 0, 'replace': 0, 'unfilled': 0}
        for i, line in enumerate(tqdm(lines, desc=f'seed {seed}')):
            assert len(line) == 3, f'line {i + 1} is not a same-identity pair: {line}'
            name, probe, a = line[0], int(line[1]), int(line[2])
            b, s_b, n_c = find_B(name, probe, a, f'{seed}-{i}')
            reason = '' if b is not None else ('only_2_images' if n_c == 0 else 'clean_check_failed')

            if b is not None:
                row = dict(idx=i, source='orig', log_idx=i, name=name, probe_id=probe, A_id=a, B_id=b,
                           replaced_reason='', replaced_pair='')
                s_a = {}
                stats['orig'] += 1
            elif args.no_replace:
                row = dict(idx=i, source='none', log_idx='', name=name, probe_id=probe, A_id=a, B_id='NA',
                           replaced_reason=reason, replaced_pair='')
                s_a, s_b = {}, {}
                stats['unfilled'] += 1
            else:
                rep = next_replacement()
                if rep is None:
                    row = dict(idx=i, source='none', log_idx='', name=name, probe_id=probe, A_id=a, B_id='NA',
                               replaced_reason=reason + '|pool_exhausted', replaced_pair='')
                    s_a, s_b = {}, {}
                    stats['unfilled'] += 1
                else:
                    r_name, r_probe, r_a, r_b, s_a, s_b = rep
                    used_people.add(r_name)
                    row = dict(idx=i, source='replace', log_idx=len(rep_lines), name=r_name, probe_id=r_probe,
                               A_id=r_a, B_id=r_b, replaced_reason=reason,
                               replaced_pair=f'{name} {probe} {a}')
                    rep_lines.append(f'{r_name}\t{r_probe}\t{r_a}')
                    stats['replace'] += 1

            for m in args.check_models:
                row[f'clean_sim_A_{m}'] = round(s_a[m], 4) if m in s_a else ''
                row[f'clean_sim_B_{m}'] = round(s_b[m], 4) if m in s_b else ''
            info.append(row)
            out_lines.append(f"{row['name']}\t{row['probe_id']}\t{row['B_id']}")

        with open(os.path.join(args.out_dir, f'transfer_same_seed_{seed}.txt'), 'w') as f:
            f.write(f'transfer_same_seed_{seed}\n' + '\n'.join(out_lines) + '\n')
        with open(os.path.join(args.out_dir, f'transfer_same_seed_{seed}_info.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(info[0].keys()))
            w.writeheader()
            w.writerows(info)
        rep_path = os.path.join(args.out_dir, f'same_seed_{seed}_replace.txt')
        with open(rep_path, 'w') as f:
            f.write(f'same_seed_{seed}_replace\n' + ''.join(l + '\n' for l in rep_lines))

        print(f"seed {seed}: {stats['orig']} original pairs kept, {stats['replace']} replaced, "
              f"{stats['unfilled']} unfilled -> {len(rep_lines)} pairs to attack in {rep_path}")


if __name__ == '__main__':
    main()