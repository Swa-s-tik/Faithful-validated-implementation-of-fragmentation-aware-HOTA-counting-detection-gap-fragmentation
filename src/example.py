"""Qualitative example (Fig. 1, graphical abstract): one real ground-truth track, faithful vs PoseTrack21 fragments.

Selection rule (fixed before looking at any track): in the confirmatory output with the largest FA-HOTA gap
(DanceTrack val, YOLOX detections, OC-SORT, raw), at alpha = 0.5, among ground-truth tracks whose true positives
are all matched to a single tracker id (so PoseTrack21 sees exactly one fragment), take the track with the most
faithful fragments; ties go to the shorter track. -> results/example.json
Usage: .venv/bin/python src/example.py
"""
import json

import numpy as np

from datasets import ROOT
from evaluate import dataset_obj
from fahota import EPS, FAHOTA

DATASET, OUTPUT, ALPHA = 'dancetrack', 'YOLOX__ocsort__raw', 0.5


def main():
    ds = dataset_obj(DATASET, ROOT / 'results/tracks' / DATASET, [OUTPUT])
    best = None
    for seq in ds.seq_list:
        data = ds.get_preprocessed_seq_data(ds.get_raw_seq_data(OUTPUT, seq), 'pedestrian')
        (t, g, p, s), *_ = FAHOTA._matches(data)
        m = s >= ALPHA - EPS
        t, g, p = t[m], g[m], p[m]
        for gid in np.unique(g):
            k = g == gid
            if len(np.unique(p[k])) != 1:
                continue
            pid = p[k][0]
            gt_frames = [f for f, ids in enumerate(data['gt_ids']) if gid in ids]
            trk_frames = [f for f, ids in enumerate(data['tracker_ids']) if pid in ids]
            tp = set(t[k].tolist())
            events = sorted(set(gt_frames) | set(trk_frames))  # frames where g or p is present
            frags, cur = [], []
            for f in events:  # faithful rule: any non-TPA event between two TPAs ends the fragment
                if f in tp:
                    cur.append(f)
                elif cur:
                    frags.append(cur)
                    cur = []
            if cur:
                frags.append(cur)
            den = len(gt_frames) + len(trk_frames) - len(tp)
            cand = dict(seq=seq, gt_id=int(gid), tracker_id=int(pid), n_gt_frames=len(gt_frames),
                        n_tp=len(tp), n_fragments_faithful=len(frags), n_fragments_pt21=1,
                        fragment_sizes=[len(f) for f in frags], A=len(tp) / den,
                        F_faithful_mean=sum(len(f) ** 2 for f in frags) / len(tp) / den,
                        F_pt21=len(tp) / den, gt_frames=gt_frames, tp_frames=sorted(tp), trk_frames=trk_frames)
            key = (len(frags), -len(gt_frames))
            if best is None or key > best[0]:
                best = (key, cand)
    ex = best[1]
    (ROOT / 'results/example.json').write_text(json.dumps(ex))
    print({k: v for k, v in ex.items() if not k.endswith('frames') and k != 'fragment_sizes'})


if __name__ == '__main__':
    main()
