"""TrackEval (MotChallenge2DBox) with FAHOTACompare + CLEAR + Identity + Count on one tracker output folder.

Writes results/metrics/<dataset>/<output>.json: per-sequence per-alpha HOTA-family quantities and CLEAR/Identity
counts (everything needed to recombine any multiset of sequences), plus TrackEval's own COMBINED_SEQ summary.
Usage: .venv/bin/python src/evaluate.py <dataset> <output_dir_name> [...]    (tracks under results/tracks/<dataset>/)
"""
import json
import sys

import numpy as np

from datasets import ROOT, SUFFIX, gt_tree
from fahota import FAHOTACompare
import trackeval

ALPHA_FIELDS = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP', 'AssA', 'FragA', 'FAssA', 'FragA_PT21', 'FAssA_PT21frag',
                'FragA_FNAonly', 'FragA_FPAonly', 'FragA_floor', 'HOTA', 'DetA', 'FA-HOTA', 'FA-HOTA_PT21',
                'FA-HOTA_PT21frag']
COUNT_FIELDS = {'CLEAR': ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'Frag', 'MT', 'PT', 'ML'],
                'Identity': ['IDTP', 'IDFN', 'IDFP'], 'Count': ['Dets', 'GT_Dets', 'IDs', 'GT_IDs']}
QUIET = dict(USE_PARALLEL=False, BREAK_ON_ERROR=True, LOG_ON_ERROR=None, PRINT_CONFIG=False, PRINT_RESULTS=False,
             TIME_PROGRESS=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
             DISPLAY_LESS_PROGRESS=True)


def dataset_obj(dataset, trackers_folder, outputs):
    gt, seqmap = gt_tree(dataset)
    return trackeval.datasets.MotChallenge2DBox(dict(
        GT_FOLDER=str(gt), SEQMAP_FILE=str(seqmap), SKIP_SPLIT_FOL=True, TRACKERS_FOLDER=str(trackers_folder),
        TRACKERS_TO_EVAL=list(outputs), TRACKER_SUB_FOLDER='', BENCHMARK='MOT17', SPLIT_TO_EVAL='train',
        CLASSES_TO_EVAL=['pedestrian'], DO_PREPROC=True, PRINT_CONFIG=False))


def flat(r):
    out = {f: np.asarray(r['FAHOTACompare'][f], float).tolist() for f in ALPHA_FIELDS}
    for m, fs in COUNT_FIELDS.items():
        out.update({f: float(r[m][f]) for f in fs})
    return out


def evaluate(dataset, outputs, trackers_folder=None, out_dir=None):
    trackers_folder = trackers_folder or ROOT / f'results/tracks{SUFFIX}' / dataset
    out_dir = out_dir or ROOT / f'results/metrics{SUFFIX}' / dataset  # SUFFIX: see datasets.ADDED
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = dataset_obj(dataset, trackers_folder, outputs)
    quiet = {'PRINT_CONFIG': False}
    metrics = [FAHOTACompare(), trackeval.metrics.CLEAR(quiet), trackeval.metrics.Identity(quiet)]  # + Count, auto
    res = trackeval.Evaluator(QUIET).evaluate([ds], metrics)[0]['MotChallenge2DBox']
    for name in outputs:
        r = {s: v['pedestrian'] for s, v in res[name].items()}
        comb = r.pop('COMBINED_SEQ')
        (out_dir / f'{name}.json').write_text(json.dumps(dict(per_seq={s: flat(v) for s, v in r.items()},
                                                              combined=flat(comb))))
        print(name, 'HOTA %.2f FragA %.2f FragA_PT21 %.2f FA-HOTA %.2f FA-HOTA_PT21 %.2f' % tuple(
            100 * np.mean(comb['FAHOTACompare'][f]) for f in ('HOTA', 'FragA', 'FragA_PT21', 'FA-HOTA',
                                                               'FA-HOTA_PT21')), flush=True)


if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2:])
