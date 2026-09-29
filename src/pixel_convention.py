"""Pixel convention of the ground truth and of every detection source, measured from the data.

1. Ground truth: boxes clamped at the image border show the convention. 0-based continuous coordinates give x = 0 and
   x + w = W at the left and right edges; MOT's 1-based convention gives x = 1 and x + w = W + 1.
2. Detections: each detection source is shifted by an offset d (added to x1, y1, x2, y2) and matched to the ground
   truth frame by frame (greedy by IoU, IoU >= 0.5, every 5th frame); the offset that maximises the mean IoU and the
   share of matches with IoU >= 0.9 is the source's offset relative to the ground truth.
-> results/pixel_convention.json.  Usage: .venv/bin/python src/pixel_convention.py
"""
import json

import numpy as np

from datasets import DATASETS, ROOT, detections, seq_info

OFFSETS = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
STRIDE = 5


def gt_boxes(dataset, seq, evaluated_only=True):
    g = np.loadtxt(DATASETS[dataset]['seq_dir'](seq) / 'gt/gt.txt', delimiter=',', ndmin=2)
    if dataset == 'mot17' and evaluated_only:
        g = g[(g[:, 6] == 1) & (g[:, 7] == 1)]  # the pedestrians the evaluator keeps
    return g


def border(dataset, evaluated_only):
    """Counts of boxes touching the left/right image border. On MOT17, all rows include static people, vehicles and
    distractors, which often touch the frame edge; the evaluated pedestrians rarely do, so their counts are too small
    to show the convention."""
    c = dict(x_eq_0=0, x_eq_1=0, right_eq_W=0, right_eq_W1=0, boxes=0)
    for seq in DATASETS[dataset]['seqs']():
        info = seq_info(dataset, seq)
        g = gt_boxes(dataset, seq, evaluated_only)
        x, w = g[:, 2], g[:, 4]
        c['boxes'] += len(g)
        c['x_eq_0'] += int((x == 0).sum())
        c['x_eq_1'] += int((x == 1).sum())
        c['right_eq_W'] += int((x + w == info['width']).sum())
        c['right_eq_W1'] += int((x + w == info['width'] + 1).sum())
    return c


def iou(a, b):
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def offset_scan(dataset, detector, min_score=0.5):
    res = {d: [] for d in OFFSETS}
    for seq in DATASETS[dataset]['seqs']():
        g = gt_boxes(dataset, seq)
        dets = detections(dataset, detector, seq)
        for f in range(1, len(dets) + 1, STRIDE):
            gg = g[g[:, 0] == f]
            dd = dets[f - 1]
            dd = dd[dd[:, 4] >= min_score]
            if not len(gg) or not len(dd):
                continue
            gb = np.c_[gg[:, 2:4], gg[:, 2:4] + gg[:, 4:6]]
            for off in OFFSETS:
                m = iou(dd[:, :4] + off, gb)
                taken_d, taken_g = set(), set()
                for k in np.argsort(-m, axis=None):
                    i, j = divmod(k, m.shape[1])
                    if m[i, j] < 0.5:
                        break
                    if i in taken_d or j in taken_g:
                        continue
                    taken_d.add(i)
                    taken_g.add(j)
                    res[off].append(m[i, j])
    out = {}
    for off, v in res.items():
        v = np.array(v)
        out[str(off)] = dict(matches=len(v), mean_iou=float(v.mean()), share_iou_ge_0_9=float((v >= 0.9).mean()))
    best = max(OFFSETS, key=lambda o: out[str(o)]['mean_iou'])
    return dict(scan=out, best_offset_mean_iou=best,
                best_offset_high_iou=max(OFFSETS, key=lambda o: out[str(o)]['share_iou_ge_0_9']))


def main():
    out = dict(ground_truth={d: dict(all_rows=border(d, False), evaluated_rows=border(d, True))
                             for d in ('mot17', 'dancetrack')}, detections={})
    for dataset, dets in (('mot17', ['FRCNN', 'SDP', 'YOLOX']), ('dancetrack', ['YOLOX', 'yolo26s', 'yolo26x'])):
        for det in dets:
            try:
                out['detections'][f'{dataset}/{det}'] = offset_scan(dataset, det)
            except FileNotFoundError as e:
                out['detections'][f'{dataset}/{det}'] = dict(error=str(e))
            r = out['detections'][f'{dataset}/{det}']
            print(dataset, det, {k: (v['mean_iou'], v['share_iou_ge_0_9']) for k, v in r.get('scan', {}).items()},
                  'best', r.get('best_offset_mean_iou'), r.get('best_offset_high_iou'), flush=True)
    print(out['ground_truth'])
    (ROOT / 'results/pixel_convention.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
