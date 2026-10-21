"""Fig. 4: PromptMIA vs naive injection, Advantage and ASR vs batch size, averaged over the three backbones.
Reads runs/fig4<tag>/<dataset>_<backbone>/attack.json; writes figures/fig4<tag>.{pdf,png} and figures/fig4<tag>_results.json."""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
p = argparse.ArgumentParser()
p.add_argument('--tag', default='')
TAG = p.parse_args().tag
DATASETS = [('cifar10', 'CIFAR10'), ('cifar100', 'CIFAR100'), ('tinyimagenet', 'TinyImageNet'), ('fourdataset', 'FourDataset')]
BACKBONES = ['vit_b32', 'deit_b16', 'convit_base']
STYLE = {('promptmia', 'advantage'): dict(color='blue', ls='-', marker='o', label='PromptMIA - Advantage'),
         ('promptmia', 'asr'): dict(color='blue', ls='--', marker='^', label='PromptMIA - Attack Success Rate'),
         ('naive', 'advantage'): dict(color='red', ls='-', marker='s', label='Naive - Advantage'),
         ('naive', 'asr'): dict(color='red', ls='--', marker='d', label='Naive - Attack Success Rate')}

table = {}
fig, axs = plt.subplots(1, 4, figsize=(20, 4.2))
for ax, (ds, title) in zip(axs, DATASETS):
    runs = {b: json.load(open(os.path.join(ROOT, 'runs', 'fig4' + TAG, f'{ds}_{b}', 'attack.json'))) for b in BACKBONES}
    table[ds] = {}
    for method in ('promptmia', 'naive'):
        bss = sorted(int(b) for b in runs[BACKBONES[0]][method])
        table[ds][method] = {}
        for metric in ('tpr', 'fpr', 'advantage', 'asr'):
            per_model = np.array([[np.mean(runs[b][method][str(bs)][metric]) for bs in bss] for b in BACKBONES])  # mean over reps
            table[ds][method][metric] = {'batch_sizes': bss, 'mean_over_models': per_model.mean(0).tolist(),
                                         'per_model': dict(zip(BACKBONES, per_model.tolist()))}
            if metric in ('advantage', 'asr'):
                ax.plot(bss, 100 * per_model.mean(0), **STYLE[(method, metric)])
    ax.set_title(f'{title} (Average 3 models)')
    ax.set_xlabel('Batch Size'); ax.set_ylabel('Percentage (%)'); ax.grid(True)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('PromptMIA vs Naive - Advantage & Attack Success Rate across Datasets')
fig.tight_layout(rect=(0, 0.06, 1, 1))
os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
fig.savefig(os.path.join(ROOT, 'figures', f'fig4{TAG}.pdf'))
fig.savefig(os.path.join(ROOT, 'figures', f'fig4{TAG}.png'), dpi=150)
json.dump(table, open(os.path.join(ROOT, 'figures', f'fig4{TAG}_results.json'), 'w'), indent=1)
for ds, _ in DATASETS:
    for method in ('promptmia', 'naive'):
        t = table[ds][method]
        print(f"{ds:13s} {method:9s} B=1: ASR {t['asr']['mean_over_models'][0]:.4f} | B=256: ASR {t['asr']['mean_over_models'][-1]:.4f}"
              f" Adv {t['advantage']['mean_over_models'][-1]:.4f} | min TPR {min(min(v) for v in t['tpr']['per_model'].values()):.4f}")
