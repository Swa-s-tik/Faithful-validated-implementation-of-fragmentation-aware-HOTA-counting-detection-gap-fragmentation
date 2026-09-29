"""Run PoseTrack21's own hota.py (verbatim) through TrackEval's multi-class path on BDD100K and record what
happens. TrackEval combines classes (class- and detection-averaged) for BDD100K, so PoseTrack21's
combine_classes_det_averaged is exercised. The first N_SEQ validation sequences (sorted) are used; the combination
code path does not depend on the number of sequences. -> results/pt21_multiclass.json
Usage: .venv/bin/python src/pt21_multiclass.py
"""
import importlib.util
import json
import tempfile
import time
import traceback
from pathlib import Path

import fahota  # noqa: F401  (restores the NumPy aliases TrackEval's and PoseTrack21's hota.py use)
import trackeval
from datasets import DATA, ROOT
from evaluate import QUIET

N_SEQ = 5
D = DATA / 'published/trackeval_data/data'


def load_pt21():
    spec = importlib.util.spec_from_file_location('trackeval.metrics.pt21_hota', ROOT / 'src/third_party/posetrack21/hota.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HOTA


def main():
    gt_all = sorted((D / 'gt/bdd100k/bdd100k_val').glob('*.json'))[:N_SEQ]
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir, trk_dir = Path(tmp) / 'gt', Path(tmp) / 'trackers/qdtrack/data'
        gt_dir.mkdir(parents=True)
        trk_dir.mkdir(parents=True)
        for f in gt_all:
            (gt_dir / f.name).symlink_to(f)
            (trk_dir / f.name).symlink_to(D / 'trackers/bdd100k/bdd100k_val/qdtrack/data' / f.name)
        ds = trackeval.datasets.BDD100K(dict(GT_FOLDER=str(gt_dir), TRACKERS_FOLDER=str(Path(tmp) / 'trackers'),
                                             TRACKERS_TO_EVAL=['qdtrack'], PRINT_CONFIG=False))
        t0, out = time.time(), dict(sequences=[f.stem for f in gt_all], classes=ds.class_list,
                                    should_classes_combine=ds.should_classes_combine)
        try:
            res = trackeval.Evaluator(dict(QUIET, BREAK_ON_ERROR=True)).evaluate([ds], [load_pt21()()])[0]
            comb = res[ds.get_name()]['qdtrack']['COMBINED_SEQ']
            out.update(status='completed', combined_keys=sorted(comb))
        except Exception as e:  # noqa: BLE001  (the outcome is the finding)
            out.update(status='raised', exception=f'{type(e).__name__}: {e}',
                       traceback=[ln.replace(str(ROOT), '<package>') for ln in traceback.format_exc().splitlines()[-8:]])
        out['seconds'] = round(time.time() - t0, 1)
    (ROOT / 'results/pt21_multiclass.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
