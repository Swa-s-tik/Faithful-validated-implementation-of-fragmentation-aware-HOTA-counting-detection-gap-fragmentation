"""Datasets, detections and TrackEval GT trees. Boxes are returned in each dataset's ground-truth pixel convention:
MOT17 is 1-based (x + w reaches W + 1 at the right edge), DanceTrack is 0-based (x = 0 and x + w = W at the edges;
results/pixel_convention.json)."""
import configparser
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# FAHOTA_ADDED=1: outputs added after pre-registration (the YOLO26 DanceTrack groups) live in separate *_added trees
ADDED = bool(os.environ.get('FAHOTA_ADDED'))
SUFFIX = '_added' if ADDED else ''
# All inputs live under data/ inside the package; FAHOTA_DATA moves the whole tree, FAHOTA_MOT17_IMG only the MOT17 images.
DATA = Path(os.environ.get('FAHOTA_DATA', ROOT / 'data'))
MOT17_ANN = DATA / 'MOT17/train'  # det/gt/seqinfo for DPM, FRCNN, SDP (from MOT17.zip)
MOT17_IMG = Path(os.environ.get('FAHOTA_MOT17_IMG', MOT17_ANN))  # <MOT17_IMG>/<seq>-FRCNN/img1/*.jpg (BoT-SORT GMC)
DANCE = DATA / 'dancetrack/val'
GT_OFFSET = {'mot17': 1.0, 'dancetrack': 0.0}  # added to 0-based detector output (src/detect.py caches)
MOT17_SEQS = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']


def _dance_seqs():
    return sorted(p.name for p in DANCE.iterdir() if p.is_dir())


DATASETS = {
    'mot17': dict(seqs=lambda: MOT17_SEQS, seq_dir=lambda s: MOT17_ANN / f'{s}-FRCNN',
                  img_dir=lambda s: MOT17_IMG / f'{s}-FRCNN/img1', public=('DPM', 'FRCNN', 'SDP'), digits=6),
    'dancetrack': dict(seqs=_dance_seqs, seq_dir=lambda s: DANCE / s, img_dir=lambda s: DANCE / s / 'img1',
                       public=(), digits=8),
}


@lru_cache
def seq_info(dataset, seq):
    c = configparser.ConfigParser()
    assert c.read(DATASETS[dataset]['seq_dir'](seq) / 'seqinfo.ini'), (dataset, seq)
    s = c['Sequence']
    return dict(length=int(s['seqLength']), fps=float(s['frameRate']), width=int(s['imWidth']),
                height=int(s['imHeight']), ext=s.get('imExt', '.jpg'))


def frames(dataset, seq):
    info, d, n = seq_info(dataset, seq), DATASETS[dataset]['img_dir'](seq), DATASETS[dataset]['digits']
    out = [(f, str(d / f'{f:0{n}d}{info["ext"]}')) for f in range(1, info['length'] + 1)]
    assert Path(out[0][1]).exists() and Path(out[-1][1]).exists(), out[0][1]
    return out


def detections(dataset, detector, seq):
    """-> list over frames 1..L of (N, 5) arrays [x1, y1, x2, y2, score] in the dataset's ground-truth convention.
    Public MOT17 detections are read from det.txt; DPM scores (unbounded, -0.5..4.8) are mapped by a logistic so
    that every detector's scores lie in (0, 1) for the trackers' fixed thresholds. Others come from src/detect.py."""
    L = seq_info(dataset, seq)['length']
    if detector in DATASETS[dataset]['public']:
        d = np.loadtxt(MOT17_ANN / f'{seq}-{detector}/det/det.txt', delimiter=',', ndmin=2)
        f, b, s = d[:, 0].astype(int), np.c_[d[:, 2:4], d[:, 2:4] + d[:, 4:6]], d[:, 6]
        if detector == 'DPM':
            s = 1.0 / (1.0 + np.exp(-s))
    elif detector == 'YOLOX':  # MOTRv2's published YOLOX db (xywh, score), used as published; offset scan in
        # results/pixel_convention.json: best at 0 px on MOT17 and -0.5 px on DanceTrack
        db, rows = _det_db(), []
        key = {'dancetrack': 'DanceTrack/val/{s}/img1/{f:08d}.txt', 'mot17': 'MOT17/images/train/{s}-SDP/img1/{f:06d}.txt'}
        for k in range(1, L + 1):
            for line in db.get(key[dataset].format(s=seq, f=k), []):
                x, y, w, h, sc = map(float, line.split(','))
                rows.append([k, x, y, x + w, y + h, sc])
        d = np.array(rows, float).reshape(-1, 6)
        f, b, s = d[:, 0].astype(int), d[:, 1:5], d[:, 5]
    else:
        z = np.load(ROOT / 'cache/dets' / dataset / f'{detector}.npz')
        assert str(z['convention']) == '0-based', 'cached detections must hold raw 0-based detector boxes'
        m = z['seq'] == seq
        f, b, s = z['frame'][m], z['boxes'][m] + GT_OFFSET[dataset], z['scores'][m]
    out = np.c_[b, s]
    return [out[f == k] for k in range(1, L + 1)]


@lru_cache
def _det_db():
    import json
    return json.loads((DATA / 'published/det_db_motrv2.json').read_text())


def gt_tree(dataset):
    """TrackEval MotChallenge2DBox GT_FOLDER and SEQMAP_FILE, built once as symlinks under cache/ (data/ stays a
    pure input directory)."""
    base = ROOT / 'cache/trackeval_gt' / dataset
    seqs = DATASETS[dataset]['seqs']()
    for s in seqs:
        (base / s / 'gt').mkdir(parents=True, exist_ok=True)
        for rel in ('gt/gt.txt', 'seqinfo.ini'):
            link = base / s / rel
            if not link.exists():
                link.symlink_to(DATASETS[dataset]['seq_dir'](s) / rel)
    seqmap = base.parent / f'{dataset}_seqmap.txt'
    text = 'name\n' + '\n'.join(seqs) + '\n'
    if not seqmap.exists() or seqmap.read_text() != text:  # parallel evaluators must not see a half-written file
        seqmap.write_text(text)
    return base, seqmap
