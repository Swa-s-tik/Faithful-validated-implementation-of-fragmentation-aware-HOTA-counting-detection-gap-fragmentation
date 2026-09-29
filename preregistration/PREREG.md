# PREREG — P1: faithful vs PoseTrack21 FA-HOTA on real tracker outputs

Written 2026-09-28 before any tracker was run and before any metric was computed on any real tracker output.
Data contact before this file: MOT17 public-detection score ranges (min/max per detector, used to choose the DPM
score mapping below) and DanceTrack/MOT17 GT file formats. B11's synthetic 10%-drop results (exploratory, 2026-09-25)
are known. The SHA-256 of this file is recorded in `DEVIATIONS.md` before any run. This file is never edited after
hashing; every change is a timestamped entry in `DEVIATIONS.md`.

## Code under test
`src/fahota.py`: `FAHOTA` (faithful HOTA Sec. 8, Eqs. 32-34 of arXiv:2009.07736v2) and `FAHOTACompare`, which adds on
the same HOTA matches: `FragA_PT21` and `FA-HOTA_PT21` (PoseTrack21 hota.py @43c9b834b6: L119 fragment rule, L211
product formula), `FA-HOTA_PT21frag` (Eq. 32 with PT21 fragments), `FragA_FNAonly`, `FragA_FPAonly` (ablations of the
break rule). Correctness gates, all passing before this file was hashed: `tests/test_fahota.py`.

## Data and outputs (48 tracker outputs)
- D1 MOT17 train, 7 sequences, full length; TrackEval MotChallenge2DBox, BENCHMARK MOT17, default preprocessing
  (pedestrian class, distractor removal). Detectors: public DPM (score mapped by a logistic), FRCNN, SDP, and
  YOLO26-X (COCO weights, person class, conf 0.05, imgsz 1280, fp16).
- D2 DanceTrack val, 25 sequences; same evaluator (DanceTrack's official command uses TrackEval's MOT challenge
  script). Detectors: YOLO26-S and YOLO26-X (COCO weights, same settings).
- Trackers, stock configurations frozen in `src/track.py` CONFIGS: SORT (abewley/sort @2236dff, max_age 1,
  min_hits 3, IoU 0.3, input score >= 0.5), ByteTrack and BoT-SORT (ultralytics 8.4.138 yaml defaults, BoT-SORT with
  sparse optical-flow GMC and no ReID, buffer scaled by fps/30), OC-SORT (noahcao/OC_SORT @8462e7e, det_thresh 0.6,
  max_age 30, min_hits 3, IoU 0.3, delta_t 3, inertia 0.2).
- Post-processing: none (raw) or ByteTrack's linear interpolation `dti` (n_min 5, n_dti 20) (interp).
- Group = (dataset, detector): 8 outputs (4 trackers x raw/interp). 6 groups.

## Metrics
All HOTA-family numbers are TrackEval's: per alpha in {0.05..0.95}, combined over sequences (sums of TP/FN/FP,
TP-weighted averages of AssA, FragA, FAssA and the PT21/ablation fields), then averaged over alpha. Also IDF1, MOTA,
IDSW and CLEAR Frag (FM) from TrackEval.

## Statistics
Bootstrap over sequences within a dataset: B = 10000 resamples with replacement, seed 20260928, the same resample
indices for every output of that dataset (paired). Metrics are recombined from per-sequence per-alpha quantities
exactly as TrackEval combines sequences. 95% percentile intervals. Two-sided bootstrap p = 2 min(P(d <= 0), P(d >= 0)),
one-sided p = P(d <= 0). Holm correction within each named family below.

## Confirmatory hypotheses
- H1 (the discrepancy exists on real outputs). For each of the 48 outputs, d_F = FragA_PT21 - FragA and
  d_H = FA-HOTA_PT21 - FA-HOTA. H1 holds if the Holm-adjusted one-sided p < 0.05 for d_H in all 48 outputs
  (family: 48 tests). Magnitudes are reported with 95% intervals and against the within-group HOTA range.
- H2 (re-ranking). In each group, for each of the 28 output pairs (i, j), take the paired differences of FA-HOTA and
  of the comparator. A significant reversal is a pair whose two differences have opposite signs and both have
  Holm-adjusted two-sided p < 0.05 (family per comparator: all 168 pairs x 2 metrics). Comparators: (a) FA-HOTA_PT21,
  (b) HOTA. H2 holds if at least one significant reversal against FA-HOTA_PT21 exists. Kendall tau between the
  rankings is reported with bootstrap 95% intervals.
- H3 (known answer: gap filling). For each (dataset, detector, tracker) triple (24), DiD =
  [FragA(interp) - FragA(raw)] - [FragA_PT21(interp) - FragA_PT21(raw)]. H3 holds if DiD > 0 with Holm-adjusted
  one-sided p < 0.05 in every triple where interpolation adds at least 0.5% boxes to the raw output (triples below
  that are reported, not tested; family: the tested triples).
- H4 (analytic oracle, deterministic). Perfect tracker (GT copied) with k = round(r n) interior boxes of each
  track removed, r in {0.025, 0.05, 0.10, 0.20}, scattered (no two adjacent) or one block, seed 0, on D1 and D2 GT.
  H4 holds if at every alpha FragA equals the closed form sum_tracks sum_fragments s^2 / n / |TP| to 1e-9 and
  FragA_PT21 equals AssA to 1e-9.

## Exploratory (labelled as such in every report)
E1 decomposition of FA-HOTA_PT21 - FA-HOTA into the L211 part (FA-HOTA_PT21 - FA-HOTA_PT21frag) and the fragment-rule
part (FA-HOTA_PT21frag - FA-HOTA); FNA-only and FPA-only ablations. E2 agreement of FragA with CLEAR Frag (FM)
normalised per GT track and per TP (Spearman across outputs, within-group Kendall). E3 wall time and memory of the
faithful code against PoseTrack21's hota.py. E4 PoseTrack21's own hota.py against FragA_PT21/FA-HOTA_PT21 on every
real sequence whose PT21 fragments array fits in 4 GiB.
