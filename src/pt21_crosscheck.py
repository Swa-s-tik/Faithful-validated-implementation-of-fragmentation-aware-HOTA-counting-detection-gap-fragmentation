"""E3/E4: PoseTrack21's own hota.py (verbatim) vs FAHOTACompare on every real sequence whose PT21 `fragments` array
(alpha x gt ids x tracker ids x timesteps, int64) fits in MAX_GIB; wall time of HOTA, FAHOTA and PT21 per sequence.
Usage: .venv/bin/python src/pt21_crosscheck.py <dataset>  -> results/pt21_crosscheck/<dataset>.json
"""
import importlib.util
import json
import sys
import time

import numpy as np

from datasets import ROOT
from evaluate import dataset_obj
from fahota import FAHOTA, FAHOTACompare
import trackeval

MAX_GIB = 4.0


def load_pt21():
    spec = importlib.util.spec_from_file_location('trackeval.metrics.pt21_hota', ROOT / 'src/third_party/posetrack21/hota.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HOTA


def timed(metric, data, reps=1):
    t = time.perf_counter()
    for _ in range(reps):
        r = metric.eval_sequence(data)
    return r, (time.perf_counter() - t) / reps


def main(dataset):
    PT21 = load_pt21()
    folder = ROOT / 'results/tracks' / dataset
    outputs = sorted(p.name for p in folder.iterdir() if p.is_dir())
    ds = dataset_obj(dataset, folder, outputs)
    rows = []
    for out in outputs:
        for seq in ds.seq_list:
            data = ds.get_preprocessed_seq_data(ds.get_raw_seq_data(out, seq), 'pedestrian')
            gib = 19 * data['num_gt_ids'] * data['num_tracker_ids'] * data['num_timesteps'] * 8 / 2 ** 30
            row = dict(output=out, seq=seq, gt_ids=data['num_gt_ids'], tracker_ids=data['num_tracker_ids'],
                       timesteps=data['num_timesteps'], pt21_fragments_gib=gib)
            _, row['t_hota'] = timed(trackeval.metrics.HOTA(), data)
            _, row['t_fahota'] = timed(FAHOTA(), data)
            mine, row['t_fahota_compare'] = timed(FAHOTACompare(), data)
            if gib <= MAX_GIB:
                theirs, row['t_pt21'] = timed(PT21(), data)
                row['max_abs_fraga'] = float(np.max(np.abs(mine['FragA_PT21'] - theirs['FragA'])))
                row['max_abs_fahota'] = float(np.max(np.abs(mine['FA-HOTA_PT21'] - theirs['FA-HOTA'])))
                row['max_abs_hota'] = float(np.max(np.abs(mine['HOTA'] - theirs['HOTA'])))
            rows.append(row)
        print(out, 'done', flush=True)
    (ROOT / 'results/pt21_crosscheck').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/pt21_crosscheck' / f'{dataset}.json').write_text(json.dumps(rows, indent=0))
    ran = [r for r in rows if 't_pt21' in r]
    print(f'{dataset}: {len(ran)}/{len(rows)} sequences fit PT21 in {MAX_GIB} GiB; max |FragA diff| '
          f'{max(r["max_abs_fraga"] for r in ran):.1e}, max |FA-HOTA diff| {max(r["max_abs_fahota"] for r in ran):.1e};'
          f' largest PT21 array {max(r["pt21_fragments_gib"] for r in rows):.1f} GiB')


if __name__ == '__main__':
    main(sys.argv[1])
