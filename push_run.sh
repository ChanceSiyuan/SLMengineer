#!/bin/bash
# push_run.sh — push a local payload to the Windows SLM runner, run it,
# and pull captured outputs back.
#
# Usage:
#   ./push_run.sh <payload_file> [--hold-on]
#
# <payload_file> must live under payload/ (e.g. payload/sheet/foo_payload.npz,
# payload/lg/..., payload/gline/..., payload/ring/..., payload/tophat/...,
# payload/wgs_square/...).
# Layout mirrors both ways:
#   local  payload/<sub>/<base>_payload.npz
#     ->   remote C:\...\slm_runner\incoming\<sub>\<base>_payload.npz
#   remote C:\...\slm_runner\data\<sub>\<base>_{before,after}.bmp + _run.json
#     ->   local  data/<sub>/<base>_{before,after}.bmp + _run.json
#
# Flags:
#   --hold-on       Display payload on SLM and hold; no capture, no pull.
#   --png           Convert BMP→PNG on Windows before pulling; pull PNG only.

set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 <payload_file> [--hold-on] [--png]" >&2
    exit 1
fi

PAYLOAD="$1"
shift

HOLD_FLAG=""
PNG_MODE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --hold-on) HOLD_FLAG="--hold-on"; shift ;;
        --png) PNG_MODE=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ ! -f "${PAYLOAD}" ]; then
    echo "ERROR: payload file not found: ${PAYLOAD}" >&2
    exit 1
fi

case "${PAYLOAD}" in
    payload/*) ;;
    *) echo "ERROR: payload must be under payload/ (got: ${PAYLOAD})" >&2; exit 1 ;;
esac

REL="${PAYLOAD#payload/}"                       # e.g. sheet/foo_payload.npz
SUBDIR="$(dirname "${REL}")"                    # e.g. sheet
FILENAME="$(basename "${PAYLOAD}")"             # e.g. foo_payload.npz
BASE="${FILENAME%_payload.npz}"                 # e.g. foo
PARAMS="$(dirname "${PAYLOAD}")/${BASE}_params.json"

# Local slash form -> Windows backslash form
SUBDIR_BS="${SUBDIR//\//\\}"

SERVER_IP="199.7.140.178"
PORT="60022"
USER="Galileo"
WIN_RUNNER_FS="/C:/Users/Galileo/slm_runner"
WIN_RUNNER_BS="C:\\Users\\Galileo\\slm_runner"
SSH_CMD="ssh -p ${PORT} ${USER}@${SERVER_IP}"
SCP_CMD="scp -P ${PORT}"

REMOTE_INCOMING_BS="${WIN_RUNNER_BS}\\incoming\\${SUBDIR_BS}"
REMOTE_INCOMING_FS="${WIN_RUNNER_FS}/incoming/${SUBDIR}"
REMOTE_DATA_FS="${WIN_RUNNER_FS}/data/${SUBDIR}"
LOCAL_DATA_DIR="data/${SUBDIR}"

# runner.py uses prefix as a path under data/, so encode the subdir into the prefix.
RUN_PREFIX="${SUBDIR_BS}\\${BASE}"

echo "[1/4] Ensuring remote ${REMOTE_INCOMING_BS}\\ exists..."
${SSH_CMD} "if not exist \"${REMOTE_INCOMING_BS}\\\" mkdir \"${REMOTE_INCOMING_BS}\""

echo "[2/4] Pushing payload..."
${SCP_CMD} "${PAYLOAD}" "${USER}@${SERVER_IP}:${REMOTE_INCOMING_FS}/"
if [ -f "${PARAMS}" ]; then
    ${SCP_CMD} "${PARAMS}" "${USER}@${SERVER_IP}:${REMOTE_INCOMING_FS}/"
    echo "  pushed ${FILENAME} + $(basename "${PARAMS}")"
else
    echo "  pushed ${FILENAME} (no params.json sibling)"
fi

echo "[3/4] Triggering slmrun.bat${HOLD_FLAG:+ (${HOLD_FLAG})}..."
${SSH_CMD} "cd /d \"${WIN_RUNNER_BS}\" && slmrun.bat \
    --payload incoming\\${SUBDIR_BS}\\${FILENAME} \
    --output-prefix ${RUN_PREFIX} ${HOLD_FLAG}"

if [ -n "${HOLD_FLAG}" ]; then
    echo "[4/4] hold-on mode: skipping pull."
    echo "Done."
    exit 0
fi

if [ -n "${PNG_MODE}" ]; then
    echo "[4/5] Converting BMP→PNG on Windows..."
    ${SSH_CMD} "uv run python - <<PY
from PIL import Image
import os
import sys

data = r\"${REMOTE_DATA_FS}\"
ok = True
for tag in (\"before\", \"after\"):
    bmp = os.path.join(data, \"${BASE}_\" + tag + \".bmp\")
    png = os.path.join(data, \"${BASE}_\" + tag + \".png\")
    if not os.path.exists(bmp):
        print(f\"  {tag}: bmp not found ({bmp})\", file=sys.stderr)
        ok = False
        continue
    Image.open(bmp).save(png)
    print(
        f\"  {tag}: {os.path.getsize(bmp)//1024}KB bmp -> \"
        f\"{os.path.getsize(png)//1024}KB png\"
    )

sys.exit(0 if ok else 2)
PY"

    echo "[5/5] Pulling PNG frames + run.json into ${LOCAL_DATA_DIR}/ ..."
    mkdir -p "${LOCAL_DATA_DIR}"
    PULL_FILES=(
        "${BASE}_before.png"
        "${BASE}_after.png"
        "${BASE}_run.json"
    )
else
    echo "[4/4] Pulling BMP frames + run.json into ${LOCAL_DATA_DIR}/ ..."
    mkdir -p "${LOCAL_DATA_DIR}"
    PULL_FILES=(
        "${BASE}_before.bmp"
        "${BASE}_after.bmp"
        "${BASE}_run.json"
    )
fi

PULL_OK=1
for f in "${PULL_FILES[@]}"; do
    if ${SCP_CMD} "${USER}@${SERVER_IP}:${REMOTE_DATA_FS}/${f}" "${LOCAL_DATA_DIR}/" 2>/dev/null; then
        echo "  ${LOCAL_DATA_DIR}/${f}"
    else
        echo "  MISSING: ${f}"
        PULL_OK=0
    fi
done

echo "Done."
