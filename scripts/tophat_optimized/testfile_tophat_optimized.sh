#!/bin/bash
# testfile_tophat_optimized.sh — optimized params from issue #15 sweeps
#
# Pipeline:
#   [1/4] Local CGM compute with best params
#   [2/4] Push payload to Windows runner
#   [3/4] Trigger hardware capture
#   [4/4] Pull results (including .npy for uniformity analysis)

set -e

SERVER_IP="199.7.140.178"
PORT="60022"
USER="Galileo"
WIN_RUNNER_FS="/C:/Users/Galileo/slm_runner"
WIN_RUNNER_BS="C:\\Users\\Galileo\\slm_runner"
SSH_CMD="ssh -p ${PORT} ${USER}@${SERVER_IP}"
SCP_CMD="scp -P ${PORT}"

PREFIX="testfile_tophat_optimized"
PAYLOAD="scripts/tophat_optimized/${PREFIX}_payload.npz"
PARAMS="scripts/tophat_optimized/${PREFIX}_params.json"

# ─── Step 1: Local CGM compute ───────────────────────────────────────
echo "[1/4] Running optimized CGM locally..."
uv run python scripts/tophat_optimized/testfile_tophat_optimized.py

if [ ! -f "${PAYLOAD}" ]; then
    echo "ERROR: ${PAYLOAD} not produced" >&2
    exit 1
fi

# ─── Step 2: Push payload + params to the Windows runner ─────────────
echo ""
echo "[2/4] Pushing payload to ${SERVER_IP}..."
${SCP_CMD} "${PAYLOAD}" "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/incoming/"
${SCP_CMD} "${PARAMS}"  "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/incoming/"
echo "  payload + params pushed"

# ─── Step 3: Trigger remote runner ───────────────────────────────────
echo ""
echo "[3/4] Triggering remote hardware runner..."
${SSH_CMD} "cd /d \"${WIN_RUNNER_BS}\" && slmrun.bat \
    --payload incoming\\${PREFIX}_payload.npz \
    --output-prefix ${PREFIX}"

# ─── Step 4: Pull captured results back into local data/ ─────────────
echo ""
echo "[4/4] Pulling results into ./data/ ..."
mkdir -p data
PULL_FILES=(
    "${PREFIX}_before.npy"
    "${PREFIX}_after.npy"
    "${PREFIX}_diff.npy"
    "${PREFIX}_before.png"
    "${PREFIX}_after.png"
    "${PREFIX}_diff.png"
    "${PREFIX}_run.json"
)
for f in "${PULL_FILES[@]}"; do
    if ${SCP_CMD} "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/data/${f}" "./data/" 2>/dev/null; then
        echo "  data/${f}"
    else
        echo "  ${f} missing"
    fi
done

echo ""
echo "Done. Next: uv run python scripts/analyze_tophat_uniformity.py"
