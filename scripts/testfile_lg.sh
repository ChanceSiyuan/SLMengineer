#!/bin/bash
# testfile_lg.sh — local CGM -> push to Windows -> run hardware -> pull results
#
# Pipeline:
#   [1/4] Local:  uv run python scripts/testfile_lg.py
#                 -> produces scripts/testfile_lg_payload.npz
#                             scripts/testfile_lg_params.json
#                             scripts/testfile_lg_preview.pdf
#   [2/4] Push:   scp payload + params to C:\Users\Galileo\slm_runner\incoming\
#   [3/4] Remote: ssh triggers runner.py on the dedicated Windows runner repo
#                 (which displays the precomputed uint8 screen and captures)
#   [4/4] Pull:   scp captured data from the runner's data/ back into ./data/
#
# The Windows runner repo is assumed to exist at C:\Users\Galileo\slm_runner\
# (see windows_runner/README.md for initial setup).  This script does NOT
# sync the main SLMengineer repo -- only the small payload files go across.

set -e

SERVER_IP="199.7.140.178"
PORT="60022"
USER="Galileo"
WIN_RUNNER_FS="/C:/Users/Galileo/slm_runner"
WIN_RUNNER_BS="C:\\Users\\Galileo\\slm_runner"
# Reuse the main SLMengineer repo's .venv (already set up with slmpy + Vimba SDK)
# so the lightweight slm_runner directory does not need its own Python env.
WIN_PYTHON="C:\\Users\\Galileo\\SLMengineer\\.venv\\Scripts\\python.exe"
SSH_CMD="ssh -p ${PORT} ${USER}@${SERVER_IP}"
SCP_CMD="scp -P ${PORT}"

PREFIX="testfile_lg"
PAYLOAD="scripts/${PREFIX}_payload.npz"
PARAMS="scripts/${PREFIX}_params.json"

# ─── Step 1: Local CGM compute ───────────────────────────────────────
echo "[1/4] Running CGM locally (~100 s on 4096^2 RTX 3090)..."
uv run python scripts/testfile_lg.py

if [ ! -f "${PAYLOAD}" ]; then
    echo "ERROR: ${PAYLOAD} not produced by testfile_lg.py" >&2
    exit 1
fi

# ─── Step 2: Push payload + params to the Windows runner ─────────────
echo ""
echo "[2/4] Pushing payload to ${SERVER_IP}:${WIN_RUNNER_BS}\\incoming\\ ..."
${SCP_CMD} "${PAYLOAD}" "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/incoming/"
${SCP_CMD} "${PARAMS}"  "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/incoming/"
echo "  payload + params pushed"

# ─── Step 3: Trigger remote runner ───────────────────────────────────
# Uses slmrun.bat -> schtasks -> run_in_session1.bat so the runner
# executes in the interactive Windows session where the SLM display
# device is accessible.
echo ""
echo "[3/4] Triggering remote hardware runner (via session-1 schtasks bridge)..."
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
echo "Done."
