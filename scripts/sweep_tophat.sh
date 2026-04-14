#!/bin/bash
# sweep_tophat.sh — push each sweep payload to Windows runner, capture results
#
# Prerequisites:
#   1. Run: uv run python scripts/sweep_tophat.py [config.json]
#      -> produces scripts/sweep_tophat/{000..NNN}_payload.npz + sweep_manifest.json
#   2. Then run this script to push/run/pull each payload on hardware.
#
# Results land in data/sweep_tophat/{IDX}_{after.png, run.json}

set -e

SERVER_IP="199.7.140.178"
PORT="60022"
USER="Galileo"
WIN_RUNNER_FS="/C:/Users/Galileo/slm_runner"
WIN_RUNNER_BS="C:\\Users\\Galileo\\slm_runner"
SSH_CMD="ssh -p ${PORT} ${USER}@${SERVER_IP}"
SCP_CMD="scp -P ${PORT}"

OUT_DIR="scripts/sweep_tophat"
MANIFEST="${OUT_DIR}/sweep_manifest.json"
DATA_DIR="data/sweep_tophat"

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: ${MANIFEST} not found. Run sweep_tophat.py first." >&2
    exit 1
fi

N=$(python3 -c "import json; print(len(json.load(open('${MANIFEST}'))))")
echo "Found ${N} sweep payloads in ${OUT_DIR}/"
echo ""

mkdir -p "${DATA_DIR}"

for i in $(seq 0 $((N-1))); do
    IDX=$(printf "%03d" "$i")
    PAYLOAD="${OUT_DIR}/${IDX}_payload.npz"
    PREFIX="sweep_tophat_${IDX}"

    if [ ! -f "${PAYLOAD}" ]; then
        echo "[${IDX}] SKIP — ${PAYLOAD} not found"
        continue
    fi

    # Extract sweep info from manifest for display
    INFO=$(python3 -c "
import json
m = json.load(open('${MANIFEST}'))[$i]
print(f\"{m['sweep_param']}={m['sweep_value']}\")
")
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[${IDX}/${N}] ${INFO}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Write a per-payload params JSON for the runner
    python3 -c "
import json
m = json.load(open('${MANIFEST}'))[$i]
params = {
    'runner_defaults': m['runner_defaults'],
    'sweep_param': m['sweep_param'],
    'sweep_value': m['sweep_value'],
    'index': m['index'],
}
json.dump(params, open('${OUT_DIR}/${IDX}_params.json', 'w'), indent=2)
"

    # Push payload + params
    echo "  push..."
    ${SCP_CMD} "${PAYLOAD}" "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/incoming/${PREFIX}_payload.npz"
    ${SCP_CMD} "${OUT_DIR}/${IDX}_params.json" "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/incoming/${PREFIX}_params.json"

    # Trigger runner
    echo "  run..."
    ${SSH_CMD} "cd /d \"${WIN_RUNNER_BS}\" && slmrun.bat \
        --payload incoming\\${PREFIX}_payload.npz \
        --output-prefix ${PREFIX}"

    # Pull results (including .npy for uniformity analysis)
    echo "  pull..."
    for f in "${PREFIX}_after.npy" "${PREFIX}_before.npy" "${PREFIX}_after.png" "${PREFIX}_run.json"; do
        if ${SCP_CMD} "${USER}@${SERVER_IP}:${WIN_RUNNER_FS}/data/${f}" "${DATA_DIR}/" 2>/dev/null; then
            echo "    ${DATA_DIR}/${f}"
        else
            echo "    ${f} missing"
        fi
    done

    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Sweep complete: ${N} payloads run.  Results in ${DATA_DIR}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
