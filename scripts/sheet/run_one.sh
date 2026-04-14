#!/bin/bash
# Drive ONE sweep point: generate (if needed) → push → run → pull
# → analyze.  Reuses data/sweep_sheet/sweep_sheet_000_before.npy as
# the common SLM-blank reference so we only pull the per-point _after
# frames (halves pull bandwidth).
#
# Usage:  ./scripts/sheet/run_one.sh <idx>
#
# Expects scripts/sweep_sheet/sweep_manifest.json maintained by
# scripts/sheet/sweep_one.py.

set -e
IDX_INT="${1:?usage: $0 <idx>}"
IDX=$(printf "%03d" "${IDX_INT}")
PREFIX="sweep_sheet_${IDX}"

SERVER_IP="199.7.140.178"
PORT="60022"
USER="Galileo"
WIN_RUNNER_FS="/C:/Users/Galileo/slm_runner"
WIN_RUNNER_BS="C:\\Users\\Galileo\\slm_runner"
SSH_CMD="ssh -p ${PORT} ${USER}@${SERVER_IP}"
SCP_CMD="scp -P ${PORT}"

OUT_DIR="scripts/sweep_sheet"
DATA_DIR="data/sweep_sheet"
MANIFEST="${OUT_DIR}/sweep_manifest.json"
PAYLOAD="${OUT_DIR}/${IDX}_payload.npz"
SHARED_BEFORE="${DATA_DIR}/sweep_sheet_000_before_crop.npy"
WIN_PY="C:\\Users\\Galileo\\SLMengineer\\.venv\\Scripts\\python.exe"
WIN_CROP="C:\\Users\\Galileo\\slm_runner\\crop_after.py"

mkdir -p "${DATA_DIR}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[idx=${IDX}] run_one"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Generate payload if missing + upsert manifest
uv run python scripts/sheet/sweep_one.py --index "${IDX_INT}" >/dev/null

# 2. Write per-point params JSON
python3 -c "
import json
m = [x for x in json.load(open('${MANIFEST}')) if x['index']==${IDX_INT}][0]
json.dump({
    'runner_defaults': m['runner_defaults'],
    'sweep_param': m['sweep_param'],
    'sweep_value': m['sweep_value'],
    'index': m['index'],
}, open('${OUT_DIR}/${IDX}_params.json', 'w'), indent=2)
"
INFO=$(python3 -c "
import json
m = [x for x in json.load(open('${MANIFEST}')) if x['index']==${IDX_INT}][0]
print(f\"{m['sweep_param']}={m['sweep_value']}\")
")
echo "[info] ${INFO}"

# 3. Push
echo "[push] payload + params"
${SCP_CMD} "${PAYLOAD}" "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/incoming/${PREFIX}_payload.npz"
${SCP_CMD} "${OUT_DIR}/${IDX}_params.json" "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/incoming/${PREFIX}_params.json"

# 4. Run on hardware
echo "[run] triggering slmrun.bat"
${SSH_CMD} "cd /d \"${WIN_RUNNER_BS}\" && slmrun.bat \
    --payload incoming\\${PREFIX}_payload.npz \
    --output-prefix ${PREFIX}"

# 5. Crop _after.npy on Windows to a ~300 KB npz (sheet ROI)
echo "[crop] remote crop of ${PREFIX}_after.npy (~70x smaller)"
${SSH_CMD} "\"${WIN_PY}\" \"${WIN_CROP}\" \"${WIN_RUNNER_BS}\\data\\${PREFIX}_after.npy\" \"${WIN_RUNNER_BS}\\data\\${PREFIX}_after_crop.npz\""

# 6. Pull cropped after + preview PNG + run JSON
echo "[pull] after_crop.npz + after.png + run.json"
for f in "${PREFIX}_after_crop.npz" "${PREFIX}_after.png" "${PREFIX}_run.json"; do
    if ${SCP_CMD} "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/data/${f}" "${DATA_DIR}/" 2>/dev/null; then
        echo "  ${DATA_DIR}/${f}"
    else
        echo "  MISSING: ${f}"
    fi
done

# 7. Extract crop npz -> crop npy for the analyzer
python3 -c "
import numpy as np
z = np.load('${DATA_DIR}/${PREFIX}_after_crop.npz')
np.save('${DATA_DIR}/${PREFIX}_after_crop.npy', z['crop'])
"

# 8. Analyze
echo "[analyze]"
uv run python scripts/sheet/analysis_sheet.py \
    --after  "${DATA_DIR}/${PREFIX}_after_crop.npy" \
    --before "${SHARED_BEFORE}" \
    --params scripts/sheet/testfile_sheet_params.json \
    --out    "${DATA_DIR}/${PREFIX}_analysis.json" \
    --preview "${DATA_DIR}/${PREFIX}_analysis.png"

echo "[done] idx=${IDX}"
