"""H5 (preregistration/PREREG_ADDENDUM_H5.md): random detector dropout -> ByteTrack / OC-SORT -> faithful vs PoseTrack21 FragA.
Usage: .venv/bin/python src/dropout.py run <dataset>    (tracks + metrics)
       .venv/bin/python src/dropout.py analyze          -> results/analysis/dropout.csv, dropout_summary.json
"""
import csv
import json
import sys

import numpy as np

from datasets import ROOT, DATASETS, detections

RATES, SEEDS, TRACKERS, DET = (0.05, 0.10, 0.20, 0.30), (0, 1, 2), ('bytetrack', 'ocsort'), 'YOLOX'
TRK_DIR, MET_DIR = ROOT / 'results/tracks_dropout', ROOT / 'results/metrics_dropout'


def name(r, s, trk):
    return f'{DET}-drop{r}-s{s}__{trk}__raw'


def run(dataset):
    import track
    from evaluate import evaluate
    outs = []
    for trk in TRACKERS:
        for r in RATES:
            for s in SEEDS:
                n = name(r, s, trk)
                seqs = DATASETS[dataset]['seqs']()
                outs.append(n)
                if all((TRK_DIR / dataset / n / f'{q}.txt').exists() for q in seqs):
                    continue  # already tracked (deterministic, see results/determinism.json)
                for k, seq in enumerate(seqs):
                    rng = np.random.default_rng([s, k])
                    dets = [d[rng.random(len(d)) >= r] for d in detections(dataset, DET, seq)]
                    track.write(track.run_tracker(trk, dets, dataset, seq), TRK_DIR / dataset / n / f'{seq}.txt')
        print(dataset, trk, 'tracked', flush=True)
    evaluate(dataset, outs, trackers_folder=TRK_DIR / dataset, out_dir=MET_DIR / dataset)


def analyze():
    from analyze import SUMS, TPW, B, SEED, CHUNK, combine, holm, p_one, ci
    rows = []
    for di, dataset in enumerate(('mot17', 'dancetrack')):
        base = {n: json.loads((ROOT / 'results/metrics' / dataset / f'{DET}__{t}__raw.json').read_text())
                for t in TRACKERS for n in [t]}
        drop = {p.stem: json.loads(p.read_text()) for p in (MET_DIR / dataset).glob('*.json')}
        runs = {**{f'{DET}-drop0-s0__{t}__raw': v for t, v in base.items()}, **drop}
        names = sorted(runs)
        seqs = sorted(runs[names[0]]['per_seq'])
        A = {k: np.array([[runs[n]['per_seq'][q][k] for q in seqs] for n in names])
             for k in ['HOTA_TP', 'HOTA_FN', 'HOTA_FP'] + TPW}
        S = {k: np.array([[runs[n]['per_seq'][q][k] for q in seqs] for n in names]) for k in SUMS}
        rng = np.random.default_rng(SEED + 10 + di)
        idx = rng.integers(0, len(seqs), size=(B, len(seqs)))
        W = np.zeros((B, len(seqs)))
        np.add.at(W, (np.arange(B)[:, None], idx), 1)
        parts = [combine(A, S, W[i:i + CHUNK]) for i in range(0, B, CHUNK)]
        bs = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        full = combine(A, S, np.ones((1, len(seqs))))
        col = {n: i for i, n in enumerate(names)}
        for trk in TRACKERS:
            i0 = col[f'{DET}-drop0-s0__{trk}__raw']
            for r in RATES:
                ii = [col[name(r, s, trk)] for s in SEEDS]
                row = dict(dataset=dataset, tracker=trk, rate=r, FragA_0=float(full['FragA'][0, i0]),
                           FragA_r=float(full['FragA'][0, ii].mean()),
                           FragA_floor_r=float(full['FragA_floor'][0, ii].mean()))  # floor: all fragments size 1
                for k in ('FragA', 'FragA_PT21', 'HOTA', 'AssA', 'FA-HOTA', 'FA-HOTA_PT21', 'FM_per_track', 'IDSW'):
                    row[f'D_{k}'] = float(full[k][0, i0] - full[k][0, ii].mean())
                    row[f'seed_sd_{k}'] = float(full[k][0, ii].std(ddof=1))
                d = (bs['FragA'][:, i0] - bs['FragA'][:, ii].mean(1)) - (bs['FragA_PT21'][:, i0] -
                                                                         bs['FragA_PT21'][:, ii].mean(1))
                row['DiD'] = row['D_FragA'] - row['D_FragA_PT21']
                row['DiD_lo'], row['DiD_hi'] = ci(d)
                row['p'] = float(p_one(d))
                rows.append(row)
    adj = holm([r['p'] for r in rows])
    for r, p in zip(rows, adj):
        r['p_holm'] = float(p)
    out = ROOT / 'results/analysis'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'dropout.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = dict(H5_n_cells=len(rows), H5_holds=bool(all(p < 0.05 for p in adj)))
    (out / 'dropout_summary.json').write_text(json.dumps(summ, indent=1))
    for r in rows:
        print(f"{r['dataset']:<10} {r['tracker']:<9} r={r['rate']:.2f}  dFragA {100 * r['D_FragA']:6.2f} "
              f"(sd {100 * r['seed_sd_FragA']:.2f})  dPT21 {100 * r['D_FragA_PT21']:6.2f}  dHOTA "
              f"{100 * r['D_HOTA']:6.2f}  dFM/track {r['D_FM_per_track']:6.2f}  DiD {100 * r['DiD']:.2f} "
              f"[{100 * r['DiD_lo']:.2f}, {100 * r['DiD_hi']:.2f}] p_holm {r['p_holm']:.4f}")
    print(summ)


if __name__ == '__main__':
    run(sys.argv[2]) if sys.argv[1] == 'run' else analyze()
