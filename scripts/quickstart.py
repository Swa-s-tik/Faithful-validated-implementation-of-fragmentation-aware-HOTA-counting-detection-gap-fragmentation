"""Quick start: faithful FragA / FA-HOTA next to the PoseTrack21 variant on a toy sequence with a detection gap.

One ground-truth object is tracked by one tracker identity for 8 frames, but frames 4 and 5 are missed. The identity
never changes, so a rule that starts a fragment only on an identity change sees one fragment; the definition in the
HOTA paper (Sec. 8, Eqs. 32-34) breaks the fragment at the gap. Needs no dataset.
Usage: .venv/bin/python scripts/quickstart.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fahota import FAHOTACompare  # noqa: E402

T = 8
gt = [np.array([0]) for _ in range(T)]
trk = [np.array([], int) if t in (3, 4) else np.array([0]) for t in range(T)]  # 0-based frames 3, 4 missed
data = dict(num_timesteps=T, gt_ids=gt, tracker_ids=trk, num_gt_ids=1, num_tracker_ids=1,
            num_gt_dets=sum(map(len, gt)), num_tracker_dets=sum(map(len, trk)),
            similarity_scores=[np.ones((len(g), len(p))) for g, p in zip(gt, trk)])

res = FAHOTACompare().eval_sequence(data)
for field in ('HOTA', 'AssA', 'FragA', 'FragA_PT21', 'FA-HOTA', 'FA-HOTA_PT21'):
    print(f'{field:<13} {np.mean(res[field]):.4f}')  # mean over the 19 localisation thresholds
