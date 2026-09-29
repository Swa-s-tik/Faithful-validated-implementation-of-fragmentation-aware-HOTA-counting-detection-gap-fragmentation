"""Re-score published tracker outputs shipped with TrackEval's test data (data.zip, sha256 in
checksums_inputs.sha256): MPNTrack on MOT15/16/17/20 train, CIWT on KITTI 2D box train, QDTrack on
BDD100K val. Same metrics as evaluate.py. Intervals: bootstrap over videos (MOT17's 21 entries are 7 videos x 3
detector sets, resampled as 7 clusters); every other benchmark has one entry per video.
Usage: .venv/bin/python src/rescore_published.py  -> results/published/<bench>.json
"""
import json
import re

import numpy as np

from analyze import B, CHUNK, SUMS, TPW, ci, combine
from datasets import DATA, ROOT
from evaluate import QUIET, flat
from fahota import FAHOTACompare
import trackeval

D = DATA / 'published/trackeval_data/data'
BENCHES = {
    **{f'{b}-train': (trackeval.datasets.MotChallenge2DBox, dict(
        GT_FOLDER=str(D / 'gt/mot_challenge'), TRACKERS_FOLDER=str(D / 'trackers/mot_challenge'),
        BENCHMARK=b, SPLIT_TO_EVAL='train', TRACKERS_TO_EVAL=['MPNTrack'], PRINT_CONFIG=False), ['pedestrian'])
       for b in ('MOT15', 'MOT16', 'MOT17', 'MOT20')},
    'KITTI-train': (trackeval.datasets.Kitti2DBox, dict(
        GT_FOLDER=str(D / 'gt/kitti/kitti_2d_box_train'), TRACKERS_FOLDER=str(D / 'trackers/kitti/kitti_2d_box_train'),
        TRACKERS_TO_EVAL=['CIWT'], PRINT_CONFIG=False), ['car', 'pedestrian']),
    'BDD100K-val': (trackeval.datasets.BDD100K, dict(
        GT_FOLDER=str(D / 'gt/bdd100k/bdd100k_val'), TRACKERS_FOLDER=str(D / 'trackers/bdd100k/bdd100k_val'),
        TRACKERS_TO_EVAL=['qdtrack'], PRINT_CONFIG=False), None),
}


def cluster_bootstrap(A, S, seqs, seed):
    """Resample videos, not sequence entries: MOT17 train lists each of its 7 videos once per public detector
    (21 entries with the same ground truth), so the three copies of a video move together."""
    video = [re.sub(r'-(DPM|FRCNN|SDP)$', '', q) for q in seqs]
    uniq = sorted(set(video))
    col = np.array([uniq.index(v) for v in video])
    rng = np.random.default_rng(seed)
    counts = np.zeros((B, len(uniq)))
    np.add.at(counts, (np.arange(B)[:, None], rng.integers(0, len(uniq), size=(B, len(uniq)))), 1)
    W = counts[:, col]
    parts = [combine(A, S, W[i:i + CHUNK]) for i in range(0, B, CHUNK)]
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}, len(uniq)


def main():
    out_dir = ROOT / 'results/published'
    out_dir.mkdir(parents=True, exist_ok=True)
    quiet = {'PRINT_CONFIG': False}
    for bench, (cls, cfg, classes) in BENCHES.items():
        ds = cls(cfg)
        classes = classes or ds.class_list
        res = trackeval.Evaluator(QUIET).evaluate(
            [ds], [FAHOTACompare(), trackeval.metrics.CLEAR(quiet), trackeval.metrics.Identity(quiet)])[0]
        tracker = cfg['TRACKERS_TO_EVAL'][0]
        r = res[ds.get_name()][tracker]
        seqs = [s for s in r if s != 'COMBINED_SEQ']
        rows = {}
        for c in classes:
            per = {s: flat(r[s][c]) for s in seqs}
            if sum(v['HOTA_TP'][0] + v['HOTA_FN'][0] for v in per.values()) == 0:
                continue  # class absent from this split
            A = {k: np.array([[per[s][k] for s in seqs]]) for k in ['HOTA_TP', 'HOTA_FN', 'HOTA_FP'] + TPW}
            S = {k: np.array([[per[s][k] for s in seqs]]) for k in SUMS if k in per[seqs[0]]}
            S.update({k: np.zeros((1, len(seqs))) for k in SUMS if k not in S})
            full = combine(A, S, np.ones((1, len(seqs))))
            bs, n_videos = cluster_bootstrap(A, S, seqs, 20260928)
            te = flat(r['COMBINED_SEQ'][c])
            assert np.isclose(full['FA-HOTA'][0, 0], np.mean(te['FA-HOTA'])), (bench, c)
            d = bs['FA-HOTA_PT21'][:, 0] - bs['FA-HOTA'][:, 0]
            rows[c] = dict(n_seqs=len(seqs), n_videos=n_videos, **{k: float(full[k][0, 0]) for k in (
                'HOTA', 'AssA', 'FragA', 'FragA_PT21', 'FA-HOTA', 'FA-HOTA_PT21', 'FA-HOTA_PT21frag')},
                dH=float(full['FA-HOTA_PT21'][0, 0] - full['FA-HOTA'][0, 0]), dH_ci=ci(d).tolist(),
                dF=float(full['FragA_PT21'][0, 0] - full['FragA'][0, 0]),
                dF_ci=ci(bs['FragA_PT21'][:, 0] - bs['FragA'][:, 0]).tolist())
            x = rows[c]
            print(f"{bench:<12} {tracker:<9} {c:<11} HOTA {100 * x['HOTA']:.2f} FragA {100 * x['FragA']:.2f} "
                  f"PT21 {100 * x['FragA_PT21']:.2f}  FA-HOTA {100 * x['FA-HOTA']:.2f} PT21 "
                  f"{100 * x['FA-HOTA_PT21']:.2f}  dH {100 * x['dH']:.2f} [{100 * x['dH_ci'][0]:.2f}, "
                  f"{100 * x['dH_ci'][1]:.2f}]", flush=True)
        (out_dir / f'{bench}.json').write_text(json.dumps(dict(tracker=tracker, classes=rows), indent=1))


if __name__ == '__main__':
    main()
