"""Paper figures from results/ (oracle JSON, analysis CSVs). Vector PDF/SVG + 300-dpi PNG in figures/.
Usage: .venv/bin/python src/figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'  # series colours: blue, orange, green
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
DS = {'mot17': 'MOT17 train', 'dancetrack': 'DanceTrack val'}
TRK = {'sort': 'SORT', 'bytetrack': 'ByteTrack', 'ocsort': 'OC-SORT', 'botsort': 'BoT-SORT'}
plt.rcParams['svg.hashsalt'] = 'fahota'  # fixed ids in SVG output, so re-runs are byte-identical
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None})  # no timestamps:
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', metadata={'Date': None})  # byte-reproducible files
    fig.savefig(OUT / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_oracle():
    """FragA vs drop rate, scattered vs one-block gaps, faithful vs PoseTrack21 (= AssA)."""
    data = {d: json.loads((ROOT / 'results/oracle' / f'{d}.json').read_text()) for d in DS
            if (ROOT / 'results/oracle' / f'{d}.json').exists()}
    fig, axes = plt.subplots(1, len(data), figsize=(3.4 * len(data), 2.4), sharey=True, squeeze=False)
    for ax, (d, res) in zip(axes[0], data.items()):
        rates = sorted({float(k.split('_')[0][1:]) for k in res})
        x = [100 * r for r in rates]
        for pat, col, mk, lab in (('block', C1, 'o', 'Faithful, one gap per track'),
                                  ('scatter', C2, 's', 'Faithful, scattered 1-frame gaps')):
            y = [100 * res[f'r{r}_{pat}']['FragA'] for r in rates]
            ax.plot(x, y, color=col, lw=2, marker=mk, ms=4, label=lab)
        y = [100 * res[f'r{r}_scatter']['FragA_PT21'] for r in rates]  # identical for both patterns (= AssA)
        ax.plot(x, y, color=C3, lw=2, marker='^', ms=4, label='PoseTrack21, both patterns')
        ax.set_title(DS[d], color=INK)
        ax.set_xlabel('Boxes removed per track (%)')
        ax.set_xticks(x)
        ax.set_ylim(0, 102)
        grid(ax)
    axes[0][0].set_ylabel('FragA (%)')
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, 'fig_oracle')


def fig_gap():
    """Per output: PoseTrack21 minus faithful, for FragA and FA-HOTA, with 95% paired bootstrap intervals."""
    o = pd.read_csv(ROOT / 'results/analysis/outputs.csv')
    o['dsn'] = o.dataset.map({d: i for i, d in enumerate(DS)})
    o = o.sort_values(['dsn', 'detector', 'tracker', 'post'], ascending=[False, False, False, False]).reset_index()
    o['label'] = [f'{DS[d].split()[0]} {a} {TRK[b]}{" +i" if c == "interp" else ""}'
                  for d, a, b, c in zip(o.dataset, o.detector, o.tracker, o.post)]
    y = list(range(len(o)))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 0.12 * len(o) + 0.9), sharey=True)
    for ax, (k, title) in zip(axes, (('dF', 'FragA: PoseTrack21 minus faithful'),
                                     ('dH', 'FA-HOTA: PoseTrack21 minus faithful'))):
        ax.errorbar(100 * o[k], y, xerr=[100 * (o[k] - o[k + '_lo']), 100 * (o[k + '_hi'] - o[k])], fmt='o', ms=3,
                    color=C1, ecolor=C1, elinewidth=0.8, capsize=0, label='PoseTrack21 minus faithful')
        if k == 'dH':
            ax.plot(100 * o['HOTA_minus_FAHOTA_PT21'], y, 's', ms=2.8, color=C2, label='HOTA minus PoseTrack21')
            ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.06 - 0.6 / len(o)), ncol=2,
                      fontsize=6)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title(title, color=INK)
        ax.set_xlabel('Points')
        grid(ax, 'x')
    for i in range(1, len(o)):  # separators between (dataset, detector) groups
        if (o.dataset[i], o.detector[i]) != (o.dataset[i - 1], o.detector[i - 1]):
            for ax in axes:
                ax.axhline(i - 0.5, color=GRID, lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(o.label, fontsize=5.5)
    axes[0].set_ylim(-0.7, len(o) - 0.3)
    save(fig, 'fig_gap')


def fig_gap_main():
    """Main-text Fig. 4: FA-HOTA gap (PoseTrack21 minus faithful) per output, larger type for double-column width."""
    o = pd.read_csv(ROOT / 'results/analysis/outputs.csv')
    o['dsn'] = o.dataset.map({d: i for i, d in enumerate(DS)})
    o = o.sort_values(['dsn', 'detector', 'tracker', 'post'], ascending=[False, False, False, False]).reset_index()
    o['label'] = [f'{DS[d].split()[0]} {a} {TRK[b]}{" +i" if c == "interp" else ""}'
                  for d, a, b, c in zip(o.dataset, o.detector, o.tracker, o.post)]
    y = list(range(len(o)))
    with plt.rc_context({'font.size': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 8, 'axes.labelsize': 9.5}):
        fig, ax = plt.subplots(figsize=(6.8, 0.17 * len(o) + 1.0))
        ax.errorbar(100 * o.dH, y, xerr=[100 * (o.dH - o.dH_lo), 100 * (o.dH_hi - o.dH)], fmt='o', ms=4, color=C1,
                    ecolor=C1, elinewidth=1.0, capsize=0, label='FA-HOTA, PoseTrack21 minus faithful (95% interval)')
        ax.plot(100 * o.HOTA_minus_FAHOTA_PT21, y, 's', ms=3.5, color=C2, label='HOTA minus PoseTrack21 FA-HOTA')
        for i in range(1, len(o)):
            if (o.dataset[i], o.detector[i]) != (o.dataset[i - 1], o.detector[i - 1]):
                ax.axhline(i - 0.5, color=GRID, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(o.label)
        ax.set_ylim(-0.7, len(o) - 0.3)
        ax.set_xlabel('Points')
        grid(ax, 'x')
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.4, -0.5 / len(o) * 8), ncol=1, fontsize=8.5)
        save(fig, 'fig_gap_main')


def fig_interp():
    """Effect of linear interpolation on FragA, faithful vs PoseTrack21, with 95% interval of the difference."""
    i = pd.read_csv(ROOT / 'results/analysis/interp.csv')
    i['label'] = [f'{DS[d].split()[0]} {a} {TRK[b]}' for d, a, b in zip(i.dataset, i.detector, i.tracker)]
    fig, ax = plt.subplots(figsize=(3.4, 0.16 * len(i) + 0.8))
    y = range(len(i))[::-1]
    ax.barh([v + 0.2 for v in y], 100 * i.dFragA, height=0.38, color=C1, label='Faithful FragA')
    ax.barh([v - 0.2 for v in y], 100 * i.dFragA_PT21, height=0.38, color=C2, label='PoseTrack21 FragA')
    ax.plot(100 * i.dAssA, y, '|', color=INK, ms=7, mew=1.2, label='AssA')
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(i.label, fontsize=6)
    ax.set_xlabel('Change from interpolation (FragA points)')
    grid(ax, 'x')
    ax.legend(frameon=False, loc='lower right', fontsize=6)
    save(fig, 'fig_interp')


def fig_dropout():
    """H5: FragA drop from random detector dropout, faithful vs PoseTrack21, per dataset and tracker."""
    f = ROOT / 'results/analysis/dropout.csv'
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    for ax, ds in zip(axes, DS):
        s = d[d.dataset == ds]
        for trk, ls in (('bytetrack', '-'), ('ocsort', '--')):
            t = s[s.tracker == trk]
            x = 100 * t.rate
            ax.plot(x, 100 * t.D_FragA, ls, color=C1, lw=2, marker='o', ms=4, label=f'Faithful FragA, {TRK[trk]}')
            ax.plot(x, 100 * t.D_FragA_PT21, ls, color=C2, lw=2, marker='s', ms=4,
                    label=f'PoseTrack21 FragA, {TRK[trk]}')
            ax.plot(x, 100 * t.D_HOTA, ls, color=C3, lw=1.5, marker='^', ms=3.5, label=f'HOTA, {TRK[trk]}')
        ax.set_title(DS[ds], color=INK)
        ax.set_xlabel('Detections dropped at random (%)')
        ax.set_xticks(sorted(set(100 * s.rate)))
        grid(ax)
    axes[0].set_ylabel('Drop from no dropout (points)')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=6)
    save(fig, 'fig_dropout')


def timeline(ax, ex, compact=False):
    """One real ground-truth track: frames matched to its tracker id, gaps, and the fragments each rule forms."""
    f0 = min(ex['gt_frames'])
    tp, gt = set(ex['tp_frames']), ex['gt_frames']
    gaps = [f - f0 for f in gt if f not in tp]
    rows = (('PoseTrack21', [ex['tp_frames']]), ('Faithful', None))
    frags, cur = [], []
    events = sorted(set(gt) | set(ex['trk_frames']))
    for f in events:
        if f in tp:
            cur.append(f)
        elif cur:
            frags.append(cur)
            cur = []
    frags += [cur] if cur else []
    for y, (label, fr) in zip((1, 0), rows):
        fr = fr or frags
        for i, seg in enumerate(fr):
            col = C1 if label == 'PoseTrack21' else (C1 if i % 2 == 0 else C3)
            ax.broken_barh([(min(seg) - f0, max(seg) - min(seg) + 1)], (y - 0.3, 0.6), facecolor=col,
                           edgecolor='white', linewidth=1.0)
        ax.text(-0.01, y, f'{label}\n{len(fr)} fragment{"s" if len(fr) > 1 else ""}', ha='right', va='center',
                fontsize=9 if compact else 7, color=INK, transform=ax.get_yaxis_transform())
    ax.vlines(gaps, -0.45, 1.45, color=C2, lw=0.8, zorder=0)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(-5, max(gt) - f0 + 5)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Frame (relative to track start)')


def fig_example():
    """Fig. 1: the selected real track (rule in src/example.py)."""
    ex = json.loads((ROOT / 'results/example.json').read_text())
    fig, ax = plt.subplots(figsize=(6.8, 1.5))
    timeline(ax, ex)
    ax.plot([], [], color=C2, lw=1, label='Ground truth present, not matched (detection gap)')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.32), fontsize=6.5)
    save(fig, 'fig_example')


if __name__ == '__main__':
    fig_example()
    fig_dropout()
    fig_oracle()
    fig_gap()
    fig_gap_main()
    fig_interp()
    print('figures ->', OUT)
