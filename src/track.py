"""Run the four stock trackers over cached detections and write MOT-format results, raw and interpolated.

results/tracks/<dataset>/<detector>__<tracker>__<raw|interp>/<seq>.txt   (frame,id,x,y,w,h,score,-1,-1,-1)
Configurations are the implementations' own defaults (frozen in CONFIGS; see preregistration/PREREG.md).
Usage: .venv/bin/python src/track.py <dataset> <detector> [<tracker> ...]
"""
import sys
import types
from pathlib import Path

import cv2
import numpy as np

from datasets import ROOT, SUFFIX, DATASETS, detections, frames, seq_info

sys.path.insert(0, str(ROOT / 'src/third_party'))
for _m in ('matplotlib', 'matplotlib.pyplot', 'matplotlib.patches', 'skimage', 'skimage.io'):  # sort.py's demo-only
    sys.modules.setdefault(_m, types.ModuleType(_m))                                          # imports, never used
sys.modules['matplotlib'].use = lambda *a, **k: None
import sort as sort_mod  # noqa: E402  (abewley/sort @2236dff, verbatim)
from ocsort.ocsort import OCSort, KalmanBoxTracker as OCKBT  # noqa: E402  (noahcao/OC_SORT @8462e7e, verbatim)
from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.trackers.bot_sort import BOTSORT  # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402

ULTRA = dict(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, track_buffer=30, match_thresh=0.8,
             fuse_score=True)  # == ultralytics 8.4.138 cfg/trackers/{bytetrack,botsort}.yaml
CONFIGS = {
    'sort': dict(max_age=1, min_hits=3, iou_threshold=0.3, det_thresh=0.5),  # sort.py defaults; det_thresh ours
    'bytetrack': dict(ULTRA, tracker_type='bytetrack'),
    'botsort': dict(ULTRA, tracker_type='botsort', gmc_method='sparseOptFlow', proximity_thresh=0.5,
                    appearance_thresh=0.8, with_reid=False, model='auto'),
    'ocsort': dict(det_thresh=0.6, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, asso_func='iou',
                   inertia=0.2, use_byte=False),  # OC_SORT MOT17 run defaults
}
INTERP = dict(n_min=5, n_dti=20)  # ByteTrack tools/interpolation.py, as its __main__ calls dti()


def run_tracker(name, dets, dataset, seq):
    """dets: per-frame (N,5) arrays. Returns rows [frame, id, x1, y1, x2, y2, score]."""
    info, cfg, rows = seq_info(dataset, seq), CONFIGS[name], []
    if name == 'sort':
        sort_mod.KalmanBoxTracker.count = 0
        trk = sort_mod.Sort(max_age=cfg['max_age'], min_hits=cfg['min_hits'], iou_threshold=cfg['iou_threshold'])
        for f, d in enumerate(dets, 1):
            for x1, y1, x2, y2, i in trk.update(d[d[:, 4] >= cfg['det_thresh']]):
                rows.append([f, int(i), x1, y1, x2, y2, 1.0])
    elif name == 'ocsort':
        OCKBT.count = 0
        trk = OCSort(**cfg)
        hw = (info['height'], info['width'])
        for f, d in enumerate(dets, 1):
            for x1, y1, x2, y2, i in trk.update(d.copy(), hw, hw):
                rows.append([f, int(i), x1, y1, x2, y2, 1.0])
    else:
        # ultralytics has no frame_rate argument: emulate original ByteTrack, buffer = fps / 30 * track_buffer
        args = IterableSimpleNamespace(**dict(cfg, track_buffer=int(info['fps'] / 30.0 * cfg['track_buffer'])))
        trk = (BYTETracker if name == 'bytetrack' else BOTSORT)(args)  # fresh tracker -> ids restart at 1
        paths = dict(frames(dataset, seq))
        for f, d in enumerate(dets, 1):
            img = cv2.imread(paths[f]) if name == 'botsort' else None  # GMC needs the frame
            assert name != 'botsort' or img is not None, paths[f]  # ultralytics skips GMC silently on None
            b = Boxes(np.hstack([d, np.zeros((len(d), 1))]).astype(np.float32), (info['height'], info['width']))
            for t in trk.update(b, img):
                rows.append([f, int(t[4]), t[0], t[1], t[2], t[3], t[5]])
    return np.array(rows, float).reshape(-1, 7)


def interpolate(rows, n_min=INTERP['n_min'], n_dti=INTERP['n_dti']):
    """ByteTrack dti(): for ids with more than n_min boxes, fill every gap 1 < frame step < n_dti linearly."""
    out = [rows]
    for i in np.unique(rows[:, 1]):
        r = rows[rows[:, 1] == i]
        r = r[np.argsort(r[:, 0])]
        if len(r) <= n_min:
            continue
        for a, b in zip(r[:-1], r[1:]):
            gap = int(b[0] - a[0])
            if 1 < gap < n_dti:
                w = (np.arange(1, gap) / gap)[:, None]
                fill = a[None] * (1 - w) + b[None] * w
                fill[:, 0], fill[:, 1] = np.arange(a[0] + 1, b[0]), i
                out.append(fill)
    rows = np.concatenate(out)
    return rows[np.lexsort((rows[:, 1], rows[:, 0]))]


def write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'{int(f)},{int(i)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.5f},-1,-1,-1\n'
                            for f, i, x1, y1, x2, y2, s in rows))


def main(dataset, detector, trackers=tuple(CONFIGS)):
    for name in trackers:
        for seq in DATASETS[dataset]['seqs']():
            rows = run_tracker(name, detections(dataset, detector, seq), dataset, seq)
            base = ROOT / f'results/tracks{SUFFIX}' / dataset
            write(rows, base / f'{detector}__{name}__raw' / f'{seq}.txt')
            write(interpolate(rows), base / f'{detector}__{name}__interp' / f'{seq}.txt')
        print(f'{dataset} {detector} {name} done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], *([sys.argv[3:]] if sys.argv[3:] else []))
