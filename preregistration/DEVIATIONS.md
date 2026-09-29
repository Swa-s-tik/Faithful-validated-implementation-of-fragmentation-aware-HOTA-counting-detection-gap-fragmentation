# Deviations from the pre-registration

`PREREG.md` and `PREREG_ADDENDUM_H5.md` are unchanged since they were hashed. This file records their hashes and every
departure from them. Times are UTC.

## Hashes

- 2026-09-28T14:27:45Z `PREREG.md`, SHA-256 `6dc7c73feb131819226509f505cceca7bde285f4631534d6175ec6a864799b7c`,
  recorded before any tracker was run and before any metric was computed on a real tracker output.
- 2026-09-28T15:14:46Z (D8) `PREREG_ADDENDUM_H5.md`, SHA-256
  `4c038323ac667c48054176630d8ea054507f5825c8d96db801a6bb87438a7a4b`, recorded before any dropout run.

## Deviations

**D1. Detectors (before any metric on real outputs).** The GPU needed for the pre-registered YOLO26 detections was not
available, so the YOLOX detections released with MOTRv2 were used on both datasets, as stored. The MOT17 YOLOX model was trained on MOT17
train, so its MOT17-train detections are in-sample. The confirmatory set is 40 outputs in 5 groups instead of 48 in 6;
the hypothesis families scale with the number of outputs. An unfinished CPU run of YOLO26 (D6, D9) produced no output.

**D2-D5. Pipeline fixes.** Duplicate field registration in `FAHOTA`, a duplicated Count metric, a race on the shared
sequence map, and DanceTrack frame names that made BoT-SORT skip camera-motion compensation. Each was fixed before the
affected numbers were read; none changed a reported number.

**D7. H2 (post hoc).** With B = 10,000 and 280 tests in the Holm family, the smallest adjusted p is 0.056, so the
pre-registered H2 test cannot pass and H2 is reported as not supported. An exploratory re-run with B = 100,000 is in
`results/analysis_B100000/`; the confirmatory results are unchanged.

**Bootstrap details.** p-values carry a +1 correction, p = (1 + #{d <= 0}) / (B + 1). Seeds are 20260928 (MOT17) and
20260929 (DanceTrack) for H1-H3, 20260938 and 20260939 for H5, and 20260928 for the published outputs; the
pre-registration names 20260928. Neither choice changes a conclusion.

**Added outputs.** YOLO26-S and YOLO26-X detections on DanceTrack val (16 outputs, recorded before they were run) are
analysed with the H1 and H3 procedures in `results/*_added/`, never pooled with the 40 confirmatory outputs.

**D10. Pixel convention.** DanceTrack ground truth is 0-based and MOT17 is 1-based (`results/pixel_convention.json`).
The cached YOLO26 DanceTrack boxes had been shifted by +1 px and were corrected. Only the 16 added outputs changed
(HOTA +0.20 to +0.59 points), and H1 (16/16) and H3 (8/8) hold on them after the fix. No confirmatory number moved.
