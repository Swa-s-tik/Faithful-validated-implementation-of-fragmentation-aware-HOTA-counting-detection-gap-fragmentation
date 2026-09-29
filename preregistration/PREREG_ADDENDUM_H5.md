# PREREG addendum H5 — detector-level dropout with real trackers

Written 2026-09-28 before any dropout run exists. Hash recorded in `DEVIATIONS.md` before the first run; never edited.
Motivation: H4 used a perfect tracker; H3 used gap filling. H5 tests the motivating mechanism directly: a detector that
misses boxes (as occlusion, low light or quantization do) feeding a real tracker.

- Detections: YOLOX (MOTRv2 db) on MOT17 train (7 seqs) and DanceTrack val (25 seqs).
- Dropout: each detection is removed independently with probability r in {0.05, 0.10, 0.20, 0.30}, seeds 0, 1, 2
  (numpy default_rng(seed) per sequence, keyed by sequence index). r = 0 is the existing raw output.
- Trackers: ByteTrack and OC-SORT, stock configs of `src/track.py`, raw output (no interpolation).
- Metrics as in PREREG.md. For each (dataset, tracker, r): the drop from r = 0, averaged over the 3 seeds,
  D_faithful = FragA(0) - mean_seed FragA(r) and D_PT21 = FragA_PT21(0) - mean_seed FragA_PT21(r).
- H5 holds if D_faithful - D_PT21 > 0 with Holm-adjusted one-sided bootstrap p < 0.05 in all 16 cells
  (2 datasets x 2 trackers x 4 rates). Bootstrap: sequences resampled (B = 10000, seed 20260928), the three seeds of a
  cell kept together (the sequence is the resampling unit). Seed-to-seed standard deviation of FragA, FragA_PT21,
  HOTA and FM is reported per cell as the seed noise.
- Reported descriptively, not tested: the same drops for HOTA, AssA, FA-HOTA, FA-HOTA_PT21 and CLEAR FM per track.
