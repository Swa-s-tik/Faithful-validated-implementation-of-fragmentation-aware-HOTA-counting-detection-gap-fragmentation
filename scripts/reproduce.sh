#!/usr/bin/env bash
# One-command reproduction of every number and figure of the paper. See README.md for data and environment setup.
# Usage: scripts/reproduce.sh            (uses cached YOLO26 detections in cache/dets if present)
#        DETECT=1 scripts/reproduce.sh   (re-runs YOLO26 detection first; needs a GPU; not guaranteed bit-identical)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$PWD/.venv/bin/python}"  # PYTHON=... selects another interpreter
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
t0=$(date +%s)

echo '== 1. correctness tests'
"$PY" tests/test_fahota.py
"$PY" tests/test_interpolation.py
"$PY" tests/mutation_check.py --out

cd src
if [[ "${DETECT:-0}" == 1 ]]; then
  echo '== 1b. YOLO26 detection (DanceTrack groups run after the confirmatory analysis)'
  for w in yolo26s yolo26x; do
    "$PY" detect.py dancetrack ../weights/$w.pt 0
  done
fi

echo '== 2. tracking (4 trackers x raw/interp) on the pre-registered detectors'
for det in DPM FRCNN SDP YOLOX; do "$PY" track.py mot17 $det; done
"$PY" track.py dancetrack YOLOX

echo '== 3. evaluation (TrackEval + faithful and PoseTrack21 FA-HOTA)'
for det in DPM FRCNN SDP YOLOX; do "$PY" evaluate.py mot17 $(cd ../results/tracks/mot17 && ls -d ${det}__*); done
"$PY" evaluate.py dancetrack $(cd ../results/tracks/dancetrack && ls -d YOLOX__*)

echo '== 4. analytic oracle (H4)'
for d in mot17 dancetrack; do "$PY" oracle.py $d; done

echo '== 5. PoseTrack21 hota.py cross-check and timing (E3, E4)'
for d in mot17 dancetrack; do "$PY" pt21_crosscheck.py $d; done

echo '== 5b. PoseTrack21 hota.py on the multi-class path (5 BDD100K sequences)'
"$PY" pt21_multiclass.py

echo '== 5c. pixel convention of ground truth and detections (D10)'
"$PY" pixel_convention.py

echo '== 6. published tracker outputs re-scored (TrackEval test data: MPNTrack, CIWT, QDTrack) (E5)'
"$PY" rescore_published.py

echo '== 7. detector dropout with real trackers (H5) and tracker determinism (E6)'
for d in mot17 dancetrack; do "$PY" dropout.py run $d; done
"$PY" determinism.py

echo '== 8. statistics: H1-H3 and E1-E2 (B = 10000), post hoc B = 100000 (D7), H5'
"$PY" analyze.py
FAHOTA_BOOT_B=100000 "$PY" analyze.py
"$PY" dropout.py analyze

echo '== 9. added outputs (YOLO26 on DanceTrack; only if cached detections exist), analysed separately'
added=()
for w in yolo26s yolo26x; do [[ -f ../cache/dets/dancetrack/$w.npz ]] && added+=($w); done
if (( ${#added[@]} )); then
  for w in "${added[@]}"; do
    FAHOTA_ADDED=1 "$PY" track.py dancetrack $w  # tracks, metrics and analysis go to the *_added trees
    FAHOTA_ADDED=1 "$PY" evaluate.py dancetrack $(cd ../results/tracks_added/dancetrack && ls -d ${w}__*)
  done
  FAHOTA_ADDED=1 "$PY" analyze.py
fi

echo '== 10. figures'
"$PY" example.py
"$PY" figures.py
"$PY" tables.py
[[ -d ../results/timing_runs ]] && "$PY" timing_summary.py  # cost table from the recorded timing runs
echo "reproduce.sh finished in $(( $(date +%s) - t0 )) s"
