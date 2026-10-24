import os
import argparse
import pickle as p

import numpy as np
import matplotlib.pyplot as plt

from utils.evolutionary_algorithms import isBetter

class history_new:
    def __init__(self, adv_score, psnr_score):
        self.adv_score = adv_score
        self.psnr_score = psnr_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_query', type=int, default=50000, help='max query number')
    parser.add_argument('--exp_dir', type=str, default='./...')
    parser.add_argument('--n_tested_imgs', type=int, default=100, help='number of images to test')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45,46], help='random seed')
    return parser.parse_args()


def get_log_dir(root_dir, method_dir, max_query, seed):
    return os.path.join(root_dir, f'{method_dir}_maxQuery-{max_query}', f'Seed{seed}', "logs")


def load_history(log_dir):
    with open(log_dir, 'rb') as haha:
        data = p.load(haha)

    history = np.asarray(data["list_adv_psnr_scores"])
    i_best = 0
    adv_best = []
    psnr_best = []
    attack_success =[]
    adv_best.append(history[0][0])
    psnr_best.append(history[0][1])
    if history[0][0] >= 0:
        attack_success.append(1)
    else:
        attack_success.append(0)
    for i in range(1, len(history)):
        A_i = history_new(history[i][0], history[i][1])
        A_i_best = history_new(history[i_best][0], history[i_best][1])
        if A_i.adv_score >= 0:
            attack_success.append(1)
        else:
            attack_success.append(0)
        if not isBetter(A_i, A_i_best):
            i_best = i
        adv_best.append(history[i_best][0])
        psnr_best.append(history[i_best][1])

    return adv_best, psnr_best, attack_success




def load_method_runs(root_dir, method_dir, max_query, seeds, n_tested_imgs):
    adv_runs = []
    psnr_runs = []
    asr_runs = []

    for seed in seeds:
        log_dir = get_log_dir(root_dir, method_dir, max_query, seed)
        adv_score = []
        psnr_score = []
        asr_score = []
        for i in range(n_tested_imgs):
            log_path = os.path.join(log_dir, f"{i}.p")
            adv_best, psnr_best, attack_success = load_history(log_path)
            adv_score.append(adv_best)
            psnr_score.append(psnr_best)
            asr_score.append(attack_success)
        adv_runs.append(adv_score)
        psnr_runs.append(psnr_score)
        asr_runs.append(asr_score)
    return adv_runs, psnr_runs, asr_runs


def aggregate_runs(runs):
    scores = []
    for run in runs:
        mean_score = np.mean(np.array(run), axis=0)
        scores.append(mean_score)
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)

    return mean_scores, std_scores


def plot_result(root_dir, max_query, seeds, n_tested_imgs):
    methods = [
        ("Combined fitness", "GA_normal", "#4C956C"),
        ("Reconstruction-bias", "GA_adaptive", "#4169E1"),
        ("Attack-bias", "GA_rules_normal", "#D95F59"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax_adv = axes[0]
    ax_psnr = axes[1]
    ax_asr = axes[2]

    queries = np.arange(1, max_query + 1)

    for label, method_dir, color in methods:
        adv_runs, psnr_runs, asr_runs = load_method_runs(root_dir, method_dir, max_query, seeds, n_tested_imgs)

        adv_mean, adv_std = aggregate_runs(adv_runs)

        psnr_mean, psnr_std = aggregate_runs(psnr_runs)

        asr_mean, asr_std = aggregate_runs(asr_runs)

        line_adv, = ax_adv.plot(queries, adv_mean, label=label, color=color)

        ax_adv.fill_between(queries, adv_mean + adv_std, adv_mean - adv_std, alpha=0.15, color=color)

        line_psnr, = ax_psnr.plot(queries, psnr_mean, label=label, color=color)

        ax_psnr.fill_between(queries, psnr_mean + psnr_std, psnr_mean - psnr_std, alpha=0.15, color=color)

        line_asr, = ax_asr.plot(queries, asr_mean, label=label, color=color)

        ax_asr.fill_between(queries, asr_mean + asr_std, asr_mean - asr_std, alpha=0.15, color=color)

    ax_adv.set_title("Adversarial Score over Queries")
    ax_adv.set_ylabel("Adversarial Score")
    ax_adv.legend()
    ax_adv.grid(True, alpha=0.3)

    ax_psnr.set_title("PSNR Score over Queries")
    ax_psnr.set_ylabel("PSNR Score")
    ax_psnr.legend()
    ax_psnr.grid(True, alpha=0.3)

    ax_asr.set_title("ASR Score over Queries")
    ax_psnr.set_xlabel("Queries")
    ax_asr.set_ylabel("ASR Score")
    ax_asr.legend()
    ax_asr.grid(True, alpha=0.3)

    ax_psnr.set_xticks(np.linspace(0, max_query, 6, dtype=int))

    plt.tight_layout()

    output_path = os.path.join(
        root_dir,
        "scores_vs_queries.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    args = parse_args()

    plot_result(
        root_dir=args.exp_dir,
        max_query=args.max_query,
        seeds=args.seeds,
        n_tested_imgs=args.n_tested_imgs
    )