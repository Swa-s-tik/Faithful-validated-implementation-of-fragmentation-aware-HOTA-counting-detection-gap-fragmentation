"""Re-run every tracker on one detector per dataset and compare with the stored outputs byte for byte.
Usage: .venv/bin/python src/determinism.py  -> results/determinism.json"""
import json
import tempfile
from pathlib import Path

import track
from datasets import ROOT, DATASETS, detections

out = {}
for dataset, det in (('mot17', 'FRCNN'), ('dancetrack', 'YOLOX')):
    for name in track.CONFIGS:
        same = True
        for seq in DATASETS[dataset]['seqs']():
            rows = track.run_tracker(name, detections(dataset, det, seq), dataset, seq)
            with tempfile.TemporaryDirectory() as tmp:
                for post, r in (('raw', rows), ('interp', track.interpolate(rows))):
                    p = Path(tmp) / f'{post}.txt'
                    track.write(r, p)
                    same &= p.read_bytes() == (ROOT / 'results/tracks' / dataset / f'{det}__{name}__{post}' /
                                               f'{seq}.txt').read_bytes()
        out[f'{dataset}/{det}/{name}'] = same
        print(dataset, det, name, 'identical' if same else 'DIFFERENT', flush=True)
(ROOT / 'results/determinism.json').write_text(json.dumps(out, indent=1))
