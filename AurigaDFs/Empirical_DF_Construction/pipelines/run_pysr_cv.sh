#!/usr/bin/env bash
#
# Run PySR over every cross-validation fold x change of variable.
#
# Expects build_cv_folds.py to have been run first, so that CV_DIR contains
# fold1/ .. foldN/ each with transforms/median_<name>.csv and test/.
#
# Usage:
#   pipelines/run_pysr_cv.sh --cv-dir <path> [options]
#
#   --cv-dir DIR        the cv_<method>_<tag>_ref<NN> directory        (required)
#   --folds "1 2 3"     which folds                                   (default: all found)
#   --transforms "..."  which changes of variable                     (default: all six)
#   --niterations N     PySR iterations                               (default: 2000)
#   --populations N     independently evolving populations            (default: 36)
#   --maxsize N         maximum complexity                            (default: 40)
#   --timeout-s N       per-run wall clock cap                        (default: none)
#   --dry-run           resolve and write configs, do not fit
#   --jobs N            run N fits concurrently                       (default: 1)
#
# Each fit writes fold<j>/pysr/<transform>/{config.json,equations.csv,result.json}
# and a line in the run log. Fits are independent, so a failure in one is reported
# and the rest continue.

set -uo pipefail

PKG_PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # AurigaDFs, i.e. the dir containing Empirical_DF_Construction
PYTHON="${PYTHON:-python}"

CV_DIR=""; FOLDS=""; TRANSFORMS=""
NITER=2000; POPS=36; MAXSIZE=40; TIMEOUT=""; DRY=""; JOBS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cv-dir)      CV_DIR="$2"; shift 2 ;;
    --folds)       FOLDS="$2"; shift 2 ;;
    --transforms)  TRANSFORMS="$2"; shift 2 ;;
    --niterations) NITER="$2"; shift 2 ;;
    --populations) POPS="$2"; shift 2 ;;
    --maxsize)     MAXSIZE="$2"; shift 2 ;;
    --timeout-s)   TIMEOUT="$2"; shift 2 ;;
    --jobs)        JOBS="$2"; shift 2 ;;
    --dry-run)     DRY="--dry-run"; shift ;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CV_DIR" ]]; then
  echo "error: --cv-dir is required" >&2; exit 2
fi
if [[ ! -f "$CV_DIR/folds.json" ]]; then
  echo "error: $CV_DIR has no folds.json -- run build_cv_folds.py first" >&2; exit 2
fi

# Default folds: whatever fold<N> directories exist, in numeric order.
if [[ -z "$FOLDS" ]]; then
  FOLDS="$(find "$CV_DIR" -maxdepth 1 -type d -name 'fold*' -exec basename {} \; \
           | sed 's/^fold//' | sort -n | tr '\n' ' ')"
fi
# Default transforms: the list the fold builder recorded, so the two cannot drift apart.
if [[ -z "$TRANSFORMS" ]]; then
  TRANSFORMS="$($PYTHON -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['transforms']))" \
                "$CV_DIR/folds.json")"
fi

if [[ -z "${FOLDS// }" ]]; then echo "error: no folds found in $CV_DIR" >&2; exit 2; fi

LOG_DIR="$CV_DIR/logs"; mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOG_DIR/run_${STAMP}.log"

echo "cv dir     : $CV_DIR"      | tee    "$SUMMARY"
echo "folds      : $FOLDS"       | tee -a "$SUMMARY"
echo "transforms : $TRANSFORMS"  | tee -a "$SUMMARY"
echo "settings   : niterations=$NITER populations=$POPS maxsize=$MAXSIZE ${TIMEOUT:+timeout=${TIMEOUT}s} jobs=$JOBS ${DRY}" \
     | tee -a "$SUMMARY"
echo "---"                       | tee -a "$SUMMARY"

run_one() {
  local fold="$1" tf="$2"
  local log="$LOG_DIR/fold${fold}_${tf}.log"
  local t0 rc
  t0=$(date +%s)
  ( cd "$PKG_PARENT" && $PYTHON -m Empirical_DF_Construction.pipelines.run_pysr_fold \
      --cv-dir "$CV_DIR" --fold "$fold" --transform "$tf" \
      --niterations "$NITER" --populations "$POPS" --maxsize "$MAXSIZE" \
      ${TIMEOUT:+--timeout-s "$TIMEOUT"} $DRY ) > "$log" 2>&1
  rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [[ $rc -eq 0 ]]; then
    echo "OK   fold${fold} ${tf} (${dt}s)"
  else
    echo "FAIL fold${fold} ${tf} (${dt}s, rc=$rc) -- see $log"
  fi
}
export -f run_one
export CV_DIR LOG_DIR PKG_PARENT PYTHON NITER POPS MAXSIZE TIMEOUT DRY

fail=0
if [[ "$JOBS" -gt 1 ]]; then
  for fold in $FOLDS; do for tf in $TRANSFORMS; do echo "$fold $tf"; done; done \
    | xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"' | tee -a "$SUMMARY"
  grep -q '^FAIL' "$SUMMARY" && fail=1
else
  for fold in $FOLDS; do
    for tf in $TRANSFORMS; do
      line="$(run_one "$fold" "$tf")"
      echo "$line" | tee -a "$SUMMARY"
      [[ "$line" == FAIL* ]] && fail=1
    done
  done
fi

echo "---"                                  | tee -a "$SUMMARY"
echo "log: $SUMMARY"                         | tee -a "$SUMMARY"
[[ $fail -eq 0 ]] && echo "all runs OK" | tee -a "$SUMMARY" || echo "some runs FAILED" | tee -a "$SUMMARY"
exit $fail
