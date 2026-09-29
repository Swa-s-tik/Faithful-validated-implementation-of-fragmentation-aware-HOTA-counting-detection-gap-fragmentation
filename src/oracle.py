"""H4: analytic oracle. A perfect tracker (the GT itself) loses k = round(r n) interior boxes of every track of n boxes,
either as k single-frame gaps (scatter, no two dropped boxes adjacent) or as one contiguous gap (block). With every
kept box identical to its GT box, each TP's F(c) = s / n, s = the size of its run of kept boxes, so
FragA = sum_tracks sum_runs s^2 / n / |TP| at every alpha, and AssA = sum_tracks (n - k)^2 / n / |TP|.
Usage: .venv/bin/python src/oracle.py <dataset>   -> results/oracle/<dataset>.json
"""
import json
import shutil
import sys

import numpy as np

from datasets import ROOT, DATASETS, gt_tree
from evaluate import dataset_obj
from fahota import FAHOTACompare
import trackeval

RATES, PATTERNS, SEED = (0.025, 0.05, 0.10, 0.20), ('scatter', 'block'), 0


def make_outputs(dataset, root):
    gt_dir, _ = gt_tree(dataset)
    rng, names, n_tot, n_drop = np.random.default_rng(SEED), [], 0, {}
    for seq in DATASETS[dataset]['seqs']():
        gt = np.loadtxt(gt_dir / seq / 'gt/gt.txt', delimiter=',', ndmin=2)
        gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]  # consider flag 1, pedestrian: what the evaluator keeps
        n_tot += len(gt)
        for r in RATES:
            keep = {p: np.ones(len(gt), bool) for p in PATTERNS}
            for tid in np.unique(gt[:, 1]):
                rows = np.flatnonzero(gt[:, 1] == tid)
                rows = rows[np.argsort(gt[rows, 0])]
                n, k = len(rows), int(round(r * len(rows)))
                if k == 0 or n - 2 < 2 * k - 1:  # interior needs room for k non-adjacent drops
                    continue
                # k non-adjacent picks among the n - 2 interior rows: sorted sample of n - 1 - k slots, spread by +i
                keep['scatter'][rows[np.sort(rng.choice(n - 1 - k, k, replace=False)) + np.arange(k) + 1]] = False
                start = rng.integers(1, n - k)
                keep['block'][rows[start:start + k]] = False
            for p, m in keep.items():
                name = f'r{r}_{p}'
                n_drop[name] = n_drop.get(name, 0) + int((~m).sum())
                (root / name).mkdir(parents=True, exist_ok=True)
                np.savetxt(root / name / f'{seq}.txt', np.c_[gt[m, :6], np.ones(m.sum()), -np.ones((m.sum(), 3))],
                           fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d')
                if name not in names:
                    names.append(name)
    return names, n_tot, n_drop


def closed_form(ds, name):
    """FragA and AssA from the preprocessed id presence alone (no matching), summed over sequences."""
    num_f = num_a = tp = 0.0
    for seq in ds.seq_list:
        d = ds.get_preprocessed_seq_data(ds.get_raw_seq_data(name, seq), 'pedestrian')
        for g in range(d['num_gt_ids']):  # perfect copy: tracker id g (contiguous) is gt id g
            present = [g in set(p) for gi, p in zip(d['gt_ids'], d['tracker_ids']) if g in set(gi)]
            n, runs, cur = len(present), [], 0
            for x in present:
                if x:
                    cur += 1
                elif cur:
                    runs.append(cur)
                    cur = 0
            runs += [cur] if cur else []
            num_f += sum(s * s for s in runs) / n
            num_a += sum(runs) ** 2 / n
            tp += sum(runs)
    return num_f / tp, num_a / tp


def main(dataset):
    root = ROOT / 'cache/oracle' / dataset
    shutil.rmtree(root, ignore_errors=True)
    names, n_tot, n_drop = make_outputs(dataset, root)
    ds = dataset_obj(dataset, root, names)
    quiet = {'PRINT_CONFIG': False}
    from evaluate import QUIET
    res = trackeval.Evaluator(QUIET).evaluate([ds], [FAHOTACompare(), trackeval.metrics.CLEAR(quiet)])[0]
    out = {}
    for name in names:
        c = res['MotChallenge2DBox'][name]['COMBINED_SEQ']['pedestrian']
        h, cl = c['FAHOTACompare'], c['CLEAR']
        fa_cf, ass_cf = closed_form(ds, name)
        out[name] = dict(dropped=n_drop[name], gt_boxes=n_tot,
                         fraga_closed=fa_cf, assa_closed=ass_cf,
                         max_abs_fraga_minus_closed=float(np.max(np.abs(h['FragA'] - fa_cf))),
                         max_abs_pt21_minus_assa=float(np.max(np.abs(h['FragA_PT21'] - h['AssA']))),
                         max_abs_assa_minus_closed=float(np.max(np.abs(h['AssA'] - ass_cf))),
                         **{f: float(np.mean(h[f])) for f in ('HOTA', 'DetA', 'AssA', 'FragA', 'FragA_PT21',
                                                               'FA-HOTA', 'FA-HOTA_PT21', 'FA-HOTA_PT21frag')},
                         CLEAR_Frag=int(cl['Frag']), IDSW=int(cl['IDSW']))
        o = out[name]
        print(f"{dataset} {name:<14} drop {100 * o['dropped'] / n_tot:5.2f}%  HOTA {100 * o['HOTA']:.2f} "
              f"AssA {100 * o['AssA']:.2f} FragA {100 * o['FragA']:.2f} (closed {100 * fa_cf:.2f}, "
              f"|d| {o['max_abs_fraga_minus_closed']:.1e}) PT21 {100 * o['FragA_PT21']:.2f} "
              f"FA-HOTA {100 * o['FA-HOTA']:.2f}/{100 * o['FA-HOTA_PT21']:.2f} FM {o['CLEAR_Frag']}", flush=True)
    (ROOT / 'results/oracle').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/oracle' / f'{dataset}.json').write_text(json.dumps(out, indent=1))
    ok = all(o['max_abs_fraga_minus_closed'] < 1e-9 and o['max_abs_pt21_minus_assa'] < 1e-9 for o in out.values())
    print('H4', 'HOLDS' if ok else 'FAILS', dataset)


if __name__ == '__main__':
    main(sys.argv[1])
