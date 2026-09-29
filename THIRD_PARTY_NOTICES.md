# Third-party notices

This repository is released under the GNU General Public License v3.0 (`LICENSE`). It vendors the third-party files
below verbatim, each under its own licence, with the licence text next to the file. Their SHA-256 sums are listed in
`checksums_inputs.sha256`.

## Vendored code (`src/third_party/`)

| Component | Files | Upstream (commit) | Licence | Licence text |
|---|---|---|---|---|
| SORT, Copyright (C) 2016-2020 Alex Bewley | `sort.py` | https://github.com/abewley/sort (`2236dff`) | GPL-3.0-or-later | `src/third_party/SORT_LICENSE` |
| OC-SORT, Copyright (c) 2021 Yifu Zhang (as stated in the upstream licence) | `ocsort/association.py`, `ocsort/kalmanfilter.py`, `ocsort/ocsort.py` | https://github.com/noahcao/OC_SORT (`8462e7e`) | MIT | `src/third_party/ocsort/LICENSE` |
| PoseTrack21 evaluation kit, Copyright (c) 2022 andoer | `posetrack21/hota.py` (the implementation compared against; used only by tests and cross-checks) | https://github.com/anDoer/PoseTrack21 (`43c9b834b6`), file `eval/posetrack21/posetrack21/trackeval/metrics/hota.py` | MIT | `src/third_party/posetrack21/LICENSE` |
| ByteTrack, Copyright (c) 2021 Yifu Zhang | `bytetrack_interpolation.py` (reference copy of `tools/interpolation.py`, used only by a test) | https://github.com/ifzhang/ByteTrack (`5d554bc`) | MIT | `src/third_party/BYTETRACK_LICENSE` |

The vendored files keep their authors' copyright notices unchanged, including the e-mail address in the SORT header.

## Dependencies (installed by pip, not redistributed)

| Package | Use here | Licence |
|---|---|---|
| TrackEval (https://github.com/JonathonLuiten/TrackEval, commit `12c8791b`) | evaluator; `src/fahota.py` subclasses its HOTA metric | MIT |
| Ultralytics 8.4.138 (https://github.com/ultralytics/ultralytics) | ByteTrack and BoT-SORT trackers; optional YOLO26 detection | AGPL-3.0 |
| PyTorch 2.14.0, torchvision 0.29.0 (CPU builds) | required by Ultralytics | BSD-3-Clause |
| NumPy, SciPy, pandas, matplotlib, OpenCV, filterpy, lap and the other pins in `requirements.lock` | numerics, I/O, figures | their own permissive licences |

Using AGPL-3.0 Ultralytics and GPL-3.0 SORT is the reason this repository is GPL-3.0.

## Data and model weights (downloaded by the user, not redistributed)

| Item | Source | Terms |
|---|---|---|
| MOT17 | https://motchallenge.net/data/MOT17/ | MOTChallenge terms (CC BY-NC-SA 3.0) |
| DanceTrack validation set | https://github.com/DanceTrack/DanceTrack | videos and images for non-commercial research only; annotations CC BY 4.0 |
| YOLOX detections released with MOTRv2 | https://github.com/megvii-research/MOTRv2 | MOTRv2 repository, MIT |
| TrackEval test data (published outputs of MPNTrack, CIWT and QDTrack with ground truth) | https://github.com/JonathonLuiten/TrackEval | the terms of the underlying datasets and tracker authors |
| YOLO26 COCO weights | https://github.com/ultralytics/assets/releases/tag/v8.4.0 | AGPL-3.0 (Ultralytics) |

`cache/dets/dancetrack/*.npz` holds detections that YOLO26 produced on DanceTrack validation frames (sequence name,
frame number, box and score only; no image content).
