#!/usr/bin/env bash
# Sweep one parameter around the locked-in light-sheet baseline (defined
# in scripts/sheet/testfile_sheet.py) and log RMS / Pk-Pk for each point.
# Per-point artifacts land under docs/sweep_sheet/param_sweep/.
#
# Usage:
#   scripts/sheet/param_sweep.sh fresnel_sd    400 600 1000 1200 1400
#   scripts/sheet/param_sweep.sh edge_sigma    0 0.5 1 2
#   scripts/sheet/param_sweep.sh flat_width    7 9 11
#   scripts/sheet/param_sweep.sh gauss_sigma   0.5 1 2
#   scripts/sheet/param_sweep.sh cgm_max_iter  5 50 500 4000

set -euo pipefail

VAR="${1:-}"; shift || true
if [ -z "$VAR" ] || [ $# -eq 0 ]; then
    echo "usage: $0 <fresnel_sd|edge_sigma|flat_width|gauss_sigma|cgm_max_iter> VAL [VAL...]" >&2
    exit 1
fi

cd /home/chance/SLMengineer

OUTDIR="docs/sweep_sheet/param_sweep"
HIST="${OUTDIR}/history.json"
mkdir -p "$OUTDIR"
[ -f "$HIST" ] || echo "[]" > "$HIST"

declare -A OVR=(
    [SLM_CGM_MAX_ITER]=4000
    [SLM_FLAT_WIDTH]=9
    [SLM_GAUSS_SIGMA]=1
    [SLM_EDGE_SIGMA]=0
    [SLM_FRESNEL_SD]=1200
    [SLM_ETIME_US]=1500
)

case "$VAR" in
    fresnel_sd)   KEY=SLM_FRESNEL_SD   ;;
    edge_sigma)   KEY=SLM_EDGE_SIGMA   ;;
    flat_width)   KEY=SLM_FLAT_WIDTH   ;;
    gauss_sigma)  KEY=SLM_GAUSS_SIGMA  ;;
    cgm_max_iter) KEY=SLM_CGM_MAX_ITER ;;
    *) echo "unknown VAR: $VAR" >&2; exit 2 ;;
esac

for V in "$@"; do
    SAFE_TAG="${VAR}_${V//./_}"
    echo ""
    echo "=== sweep ${VAR}=${V} ==="

    OVR[$KEY]="$V"
    ENV_ARGS=()
    for k in "${!OVR[@]}"; do ENV_ARGS+=("$k=${OVR[$k]}"); done

    env "${ENV_ARGS[@]}" uv run python scripts/sheet/testfile_sheet.py \
        > "${OUTDIR}/${SAFE_TAG}_build.log" 2>&1 || { echo "BUILD FAILED"; continue; }
    ./push_run.sh payload/sheet/testfile_sheet_payload.npz \
        > "${OUTDIR}/${SAFE_TAG}_push.log" 2>&1 || { echo "PUSH FAILED"; continue; }
    uv run python scripts/sheet/analysis_sheet.py \
        --plot "${OUTDIR}/${SAFE_TAG}_plot.png" \
        --result "${OUTDIR}/${SAFE_TAG}_result.json" \
        > "${OUTDIR}/${SAFE_TAG}_analyse.log" 2>&1 || { echo "ANALYSE FAILED"; continue; }

    python3 - "$VAR" "$V" "${OUTDIR}/${SAFE_TAG}_result.json" "$HIST" <<'PY'
import json, sys
var, val, result_p, hist_p = sys.argv[1:5]
r = json.load(open(result_p))
row = {
    "var": var, "value": val,
    "rms_percent": r["rms_percent"],
    "pk_pk_percent": r["pk_pk_percent"],
    "flat_top_bounds_px": r["flat_top_bounds_px"],
    "flat_top_mean_intensity": r["flat_top_mean_intensity"],
    "roi_shape_yx": r["roi_shape_yx"],
    "plot_path": r["plot_path"],
}
hist = json.load(open(hist_p))
hist.append(row)
json.dump(hist, open(hist_p, "w"), indent=2)
print(f"  {var}={val}: RMS={row['rms_percent']:.2f}%  "
      f"Pk-Pk={row['pk_pk_percent']:.2f}%  "
      f"flat={row['flat_top_bounds_px']}  mean={row['flat_top_mean_intensity']:.1f}")
PY
done

echo ""
echo "Sweep done; see ${HIST}."
