"""Stock COCO person detector (ultralytics YOLO26) over every frame of a dataset -> cache/dets/<dataset>/<det>.npz.

Boxes are stored as the detector outputs them (0-based continuous xyxy, file field convention='0-based'), with frame
numbers as in the dataset; datasets.detections() adds the dataset's ground-truth offset (MOT17 1-based: +1,
DanceTrack 0-based: 0).
Usage: .venv/bin/python src/detect.py <dataset> <weights> [device]   (fp16 only on GPU)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from datasets import DATASETS, frames

CONF, IMGSZ, PERSON = 0.05, 1280, 0  # low conf so two-stage trackers see their low-score band


def main(dataset, weights, device='0'):
    root = Path(__file__).resolve().parents[1]
    name = Path(weights).stem
    out = root / 'cache/dets' / dataset / f'{name}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    if device == 'cpu':  # OMP_NUM_THREADS alone does not cap torch/OpenCV pools
        import cv2
        import os
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
        cv2.setNumThreads(1)
    model = YOLO(weights)
    seqs, frs, boxes, scores = [], [], [], []
    t0, n = time.time(), 0
    for seq in DATASETS[dataset]['seqs']():
        for f, path in frames(dataset, seq):
            r = model.predict(path, imgsz=IMGSZ, conf=CONF, classes=[PERSON], half=device != 'cpu', device=device,
                              verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy().astype(np.float64)  # 0-based; the GT offset is added when loading
            s = r.boxes.conf.cpu().numpy().astype(np.float64)
            seqs += [seq] * len(b)
            frs += [f] * len(b)
            boxes.append(b)
            scores.append(s)
            n += 1
    np.savez_compressed(out, seq=np.array(seqs), frame=np.array(frs, int), boxes=np.concatenate(boxes),
                        scores=np.concatenate(scores), convention=np.array('0-based'))
    peak = torch.cuda.max_memory_allocated() / 2**20 if device != 'cpu' else 0  # never touch CUDA on CPU runs
    print(f'{dataset} {name}: {n} frames, {len(frs)} boxes, {time.time() - t0:.0f}s, device {device}, '
          f'peak GPU {peak:.0f} MiB -> {out}')


if __name__ == '__main__':
    main(*sys.argv[1:4])
