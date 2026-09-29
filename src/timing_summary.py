"""Cost table from the timing files of every recorded run (results/timing_runs/<run>/{mot17,dancetrack}.json, copies
of results/pt21_crosscheck/ from each full run). Per-sequence ratios of wall time to TrackEval's HOTA; single timings
on a shared machine, so the table reports every run and the spread. -> results/tables/cost.md
Usage: .venv/bin/python src/timing_summary.py"""
import json

import numpy as np

from datasets import ROOT

RUNS = ROOT / 'results/timing_runs'
DS = {'mot17': 'MOT17 train', 'dancetrack': 'DanceTrack val'}


def stats(path):
    r = json.loads(path.read_text())
    ran = [x for x in r if 't_pt21' in x]
    fa = np.median([x['t_fahota'] / x['t_hota'] for x in r])
    pt = np.array([x['t_pt21'] / x['t_hota'] for x in ran])
    return len(r), len(ran), fa, np.median(pt), pt.max(), max(x['pt21_fragments_gib'] for x in r)


def main():
    rows = ['| Run | Dataset | Sequence outputs (PoseTrack21 run) | Faithful / HOTA, median | PoseTrack21 / HOTA, median | '
            'PoseTrack21 / HOTA, max | Largest PoseTrack21 array (GiB) |', '|' + '---|' * 7]
    agg = {d: [] for d in DS}
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        for d in DS:
            n, n_pt, fa, pt, mx, gib = stats(run / f'{d}.json')
            agg[d].append((fa, pt, mx))
            rows.append(f'| {run.name} | {DS[d]} | {n} ({n_pt}) | {fa:.2f} | {pt:.1f} | {mx:.0f} | {gib:.1f} |')
    for d, v in agg.items():
        v = np.array(v)
        rows.append(f'| all runs | {DS[d]} | | {v[:, 0].min():.2f} to {v[:, 0].max():.2f} | {v[:, 1].min():.1f} to '
                    f'{v[:, 1].max():.1f} | {v[:, 2].min():.0f} to {v[:, 2].max():.0f} | |')
    (ROOT / 'results/tables').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/tables/cost.md').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
