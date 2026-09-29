"""src/track.py interpolate() against ByteTrack's own dti() (src/third_party/bytetrack_interpolation.py, verbatim).

Self-contained: ByteTrack and OC-SORT are run here on one MOT17 and one DanceTrack sequence, the raw rows are written
as MOT text files, ByteTrack's dti() is run on them, and its output is compared with interpolate() applied to the same
written rows. The vendored file imports motmetrics and yolox at module level for its unused eval_mota(); stubs replace
them here, so neither is a dependency. Run: .venv/bin/python tests/test_interpolation.py
"""
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for _m in ('motmetrics', 'yolox', 'yolox.evaluators', 'yolox.evaluators.evaluation'):  # used only by eval_mota()
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules['yolox.evaluators.evaluation'].Evaluator = None
spec = importlib.util.spec_from_file_location('bytetrack_interpolation', ROOT / 'src/third_party/bytetrack_interpolation.py')
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)
import track  # noqa: E402
from datasets import DATASETS, detections  # noqa: E402

CASES = [('mot17', 'FRCNN', 'bytetrack', 'MOT17-09'), ('dancetrack', 'YOLOX', 'ocsort', None)]


def rows_of(path):
    """MOT txt -> [frame, id, x1, y1, x2, y2] sorted by (frame, id)."""
    d = np.loadtxt(path, delimiter=',', ndmin=2)
    r = np.c_[d[:, :2], d[:, 2:4], d[:, 2:4] + d[:, 4:6]]
    return r[np.lexsort((r[:, 1], r[:, 0]))]


def test_matches_bytetrack_dti():
    for dataset, det, trk, seq in CASES:
        seq = seq or DATASETS[dataset]['seqs']()[0]
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir, out_dir = Path(tmp) / 'raw', Path(tmp) / 'dti'
            out_dir.mkdir()
            track.write(track.run_tracker(trk, detections(dataset, det, seq), dataset, seq), raw_dir / f'{seq}.txt')
            bt.dti(str(raw_dir), str(out_dir), n_min=track.INTERP['n_min'], n_dti=track.INTERP['n_dti'])
            raw = np.loadtxt(raw_dir / f'{seq}.txt', delimiter=',', ndmin=2)
            ours = track.interpolate(np.c_[raw[:, :2], raw[:, 2:4], raw[:, 2:4] + raw[:, 4:6], raw[:, 6]])[:, :6]
            ours = ours[np.lexsort((ours[:, 1], ours[:, 0]))]
            theirs = rows_of(out_dir / f'{seq}.txt')
            assert len(raw) > 100 and len(ours) > len(raw), (seq, len(raw), len(ours))  # non-vacuous: gaps filled
            assert ours.shape == theirs.shape, (seq, ours.shape, theirs.shape)
            assert np.array_equal(ours[:, :2], theirs[:, :2]), seq
            assert np.allclose(ours[:, 2:], theirs[:, 2:], atol=1e-6), seq
        print(f'PASS {dataset} {det} {trk} {seq}: {len(raw)} raw rows, {len(ours) - len(raw)} interpolated')


if __name__ == '__main__':
    test_matches_bytetrack_dti()
