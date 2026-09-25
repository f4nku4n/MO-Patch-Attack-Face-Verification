import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rc('legend', fontsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('axes', labelsize=20)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


asr_all_vggface = p.load(open('/content/drive/MyDrive/colab_notebooks/MO-Patch-Attack-Face_Verification/ASR_vggface.p', 'rb'))
asr_all_webface = p.load(open('/content/drive/MyDrive/colab_notebooks/MO-Patch-Attack-Face_Verification/ASR_webface.p', 'rb'))
asr_all_arcface = p.load(open('/content/drive/MyDrive/colab_notebooks/MO-Patch-Attack-Face_Verification/ASR_arcface.p', 'rb'))
asr_all_cosface = p.load(open('/content/drive/MyDrive/colab_notebooks/MO-Patch-Attack-Face_Verification/ASR_cosface.p', 'rb'))

all_asr = {
    'WebFace': [asr_all_webface['GA-recons'], asr_all_webface['GA-combined'], asr_all_webface['GA-attack'], asr_all_webface['Ours']],
    'VGGFace': [asr_all_vggface['GA-recons'], asr_all_vggface['GA-combined'], asr_all_vggface['GA-attack'], asr_all_vggface['Ours']],
    'ArcFace': [asr_all_arcface['GA-recons'], asr_all_arcface['GA-combined'], asr_all_arcface['GA-attack'], asr_all_arcface['Ours']],
    'CosFace': [asr_all_cosface['GA-recons'], asr_all_cosface['GA-combined'], asr_all_cosface['GA-attack'], asr_all_cosface['Ours']]

}

all_asr = {'WebFace': {}, 'VGGFace': {}, 'ArcFace': {}, 'CosFace': {}}
all_asr['WebFace'] = {
    'GA-Recons': asr_all_webface['GA-recons'],
    'GA-Combined': asr_all_webface['GA-combined'],
    'GA-Attack': asr_all_webface['GA-attack'],
    'Ours': asr_all_webface['Ours']
}
all_asr['VGGFace'] = {
    'GA-Recons': asr_all_vggface['GA-recons'],
    'GA-Combined': asr_all_vggface['GA-combined'],
    'GA-Attack': asr_all_vggface['GA-attack'],
    'Ours': asr_all_vggface['Ours']
}
all_asr['ArcFace'] = {
    'GA-Recons': asr_all_arcface['GA-recons'],
    'GA-Combined': asr_all_arcface['GA-combined'],
    'GA-Attack': asr_all_arcface['GA-attack'],
    'Ours': asr_all_arcface['Ours']
}
all_asr['CosFace'] = {
    'GA-Recons': asr_all_cosface['GA-recons'],
    'GA-Combined': asr_all_cosface['GA-combined'],
    'GA-Attack': asr_all_cosface['GA-attack'],
    'Ours': asr_all_cosface['Ours']
}

linewidth = 2.5
marker_queries = np.array([2000, 4000, 6000, 8000, 10000]) - 1
method2color = {
    'GA-Recons': 'tab:blue',
    'GA-Combined': 'tab:orange',
    'GA-Attack': 'tab:green',
    'Ours': 'tab:red'
}
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
for _ax in ax.flat:
    _ax.set_ylim(-0.05, 1.05)
    _ax.set_xticks(range(0, 10001, 2000))
    _ax.get_xaxis().set_major_formatter(FormatStrFormatter('%.d'))
    _ax.set_yticks(np.arange(0, 1.01, 0.2))
    _ax.grid(True, linestyle='--')

i, j = 0, 0
query_checkpoint = 10000
for victim_model, asr_list in all_asr.items():
    for method, list_asr_method in asr_list.items():
        mean_asr = np.mean(list_asr_method, axis=0)[:query_checkpoint]
        std_asr  = np.std(list_asr_method, axis=0)[:query_checkpoint]

        X = np.array(list(range(query_checkpoint)))
        ax[i, j].plot(X, mean_asr, linewidth=linewidth, label=method, zorder=2,
                      marker='o',
                      markevery=marker_queries,
                      markersize=8,
                      markerfacecolor='none',
                      markeredgecolor=method2color[method],
                      markeredgewidth=2)
        ax[i, j].fill_between(range(query_checkpoint), mean_asr-std_asr, mean_asr+std_asr, alpha=0.2)

    ax[i, j].set_title(f'{victim_model}', fontsize=20)
    j += 1
    if j == 2:
        j = 0
        i += 1
ax[0, 0].set_ylabel("ASR")
ax[1, 0].set_ylabel("ASR")
ax[1, 0].set_xlabel("Queries")
ax[1, 1].set_xlabel("Queries")


handles, labels = ax[0, 0].get_legend_handles_labels()

fig.legend(
    handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
    ncol=4, frameon=True
)

plt.tight_layout(rect=[0, 0.07, 1, 1])

plt.savefig(f'/content/drive/MyDrive/colab_notebooks/MO-Patch-Attack-Face_Verification/ConGaConHeo.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()