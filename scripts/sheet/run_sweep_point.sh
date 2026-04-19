#!/bin/bash
# run_sweep_point.sh — single iteration of the Issue #19 sheet sweep.
#
# Usage: run_sweep_point.sh <param> <value> <tag>
#   param = setting_eta | cgm_eta_steepness | sheet_flat_width
#         | sheet_gaussian_sigma | sheet_edge_sigma
#   value = numeric value to substitute
#   tag   = short id used in filenames, e.g. eta_0p4, flatw_15
#
# Effects:
#   1. sed the param in scripts/sheet/testfile_sheet.py to <value>.
#   2. uv run scripts/sheet/testfile_sheet.py → regenerates payload.
#   3. ./push_run.sh … --png → pushes to Windows and captures.
#   4. Archives PNGs + params + preview under docs/sweep_sheet/<tag>_*.
#   5. Runs analysis_sheet.py --peak signal and appends one row to
#      docs/sweep_sheet/_index.csv.
#
# Assumes you are in the repo root. Does NOT restore the param — caller
# must reset testfile_sheet.py between parameter groups.

set -e

if [ $# -lt 3 ]; then
    echo "usage: $0 <param> <value> <tag>" >&2
    exit 1
fi

PARAM="$1"
VALUE="$2"
TAG="$3"

SCRIPT="scripts/sheet/testfile_sheet.py"
ARCHIVE="docs/sweep_sheet"
IDX="${ARCHIVE}/_index.csv"

# --- 1. Patch the parameter line.  Each parameter has a unique leading
#        identifier so sed is unambiguous.
case "${PARAM}" in
    setting_eta|cgm_eta_steepness|sheet_flat_width|sheet_gaussian_sigma|sheet_edge_sigma|cgm_max_iterations|cgm_steepness|fresnel_sd)
        # Replace RHS up to a trailing `#` comment or EOL.
        sed -i -E "s|^(    ${PARAM} *= *)[^#]*(#.*)?\$|\1${VALUE}  \2|" "${SCRIPT}"
        echo "[patch] ${PARAM} = ${VALUE}"
        grep -nE "^    ${PARAM} *=" "${SCRIPT}" || { echo "ERROR: ${PARAM} not found"; exit 1; }
        ;;
    *)
        echo "unknown param: ${PARAM}" >&2
        exit 1
        ;;
esac

# --- 2. Generate payload.
uv run python "${SCRIPT}"

# --- 3. Push + capture + pull PNG.
./push_run.sh payload/sheet/testfile_sheet_payload.npz --png

# --- 4. Archive capture + params.
cp data/sheet/testfile_sheet_after.png  "${ARCHIVE}/${TAG}_after.png"
cp data/sheet/testfile_sheet_before.png "${ARCHIVE}/${TAG}_before.png"
cp payload/sheet/testfile_sheet_params.json "${ARCHIVE}/${TAG}_params.json"

# --- 5. Analyze + preview.
uv run python scripts/sheet/analysis_sheet.py \
    --after  data/sheet/testfile_sheet_after.png \
    --before data/sheet/testfile_sheet_before.png \
    --params payload/sheet/testfile_sheet_params.json \
    --out    "${ARCHIVE}/${TAG}_metrics.json" \
    --preview "${ARCHIVE}/${TAG}_preview.png" \
    --peak signal

# --- 6. Index row.
if [ ! -f "${IDX}" ]; then
    echo "tag,param,value,flat_width_px,gauss_sigma_px,edge_sigma_px,eff_pct,fid_corr,fid_overlap,flat_rms,sim_F,sim_eta" > "${IDX}"
fi
uv run python - "${ARCHIVE}/${TAG}_metrics.json" "${ARCHIVE}/${TAG}_params.json" "${TAG}" "${PARAM}" "${VALUE}" >> "${IDX}" <<'PY'
import json, sys
metrics_p, params_p, tag, param, value = sys.argv[1:6]
metrics = json.load(open(metrics_p))
params  = json.load(open(params_p))
fit = metrics["fit"]
m   = metrics["metrics"]

def fmt(x):
    if isinstance(x, float):
        if x != x:  # NaN
            return "nan"
        return f"{x:.4f}"
    return str(x)

row = [
    tag, param, value,
    fit.get("measured_flat_width_px", float("nan")),
    fit.get("measured_gauss_sigma_px", float("nan")),
    fit.get("measured_edge_sigma_px", float("nan")),
    100.0 * float(m.get("efficiency", 0.0)),
    m.get("fidelity_corr", float("nan")),
    m.get("fidelity_overlap", float("nan")),
    m.get("flat_region_rms", float("nan")),
    params.get("final_fidelity", float("nan")),
    params.get("final_efficiency", float("nan")),
]
print(",".join(fmt(x) for x in row))
PY

echo "[done] ${TAG}"
