#!/bin/bash
# SLM remote execution bridge
# Usage: ./ai_slm_loop.sh <script.py> [file1_to_pull] [file2_to_pull] ...
#
# Step 1: Sync repo to remote Windows (excludes .gitignore patterns + data/)
# Step 2: Run script via slmrun.bat (interactive Session 1, SLM accessible)
# Step 3: Pull specified data files into local data/ directory
#
# Examples:
#   ./ai_slm_loop.sh testfile.py testfile_before.png testfile_after.png testfile_diff.npy
#   ./ai_slm_loop.sh scan.py scan_result.json scan_data.npy

set -e

if [ -z "$1" ]; then
    echo "Usage: ./ai_slm_loop.sh <script.py> [files_to_pull ...]"
    echo "  Files are pulled into local data/ directory."
    exit 1
fi

FILE="${1#scripts/}"
shift
PULL_FILES=("$@")

SERVER_IP="199.7.140.178"
PORT="60022"
USER="Galileo"
WIN_DIR_BS="C:\Users\Galileo\SLMengineer"
WIN_DIR_FS="/C:/Users/Galileo/SLMengineer"
SSH_CMD="ssh -p $PORT ${USER}@${SERVER_IP}"

# ─── Step 1: Sync repo ───────────────────────────────────────────────
echo "[1/3] Syncing repo to Windows..."

TAR_FILE="/tmp/slm_sync_$$.tar.gz"
tar czf "$TAR_FILE" \
    --exclude='.git' \
    --exclude='.claude' \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='.idea' \
    --exclude='data' \
    --exclude='*.npy' \
    --exclude='*.png' \
    --exclude='*_done.flag' \
    --exclude='*_output.txt' \
    --exclude='*.pyc' \
    -C "$(pwd)" .
scp -P $PORT "$TAR_FILE" "${USER}@${SERVER_IP}:${WIN_DIR_FS}/_sync.tar.gz"
$SSH_CMD "cd /d \"${WIN_DIR_BS}\" && tar xzf _sync.tar.gz && del _sync.tar.gz"
rm -f "$TAR_FILE"

# ─── Step 2: Run script via slmrun.bat ───────────────────────────────
echo "[2/3] Running ${FILE} on Windows (interactive session)..."

$SSH_CMD "cd /d \"${WIN_DIR_BS}\" && slmrun.bat scripts/$FILE"

# ─── Step 3: Pull data files ─────────────────────────────────────────
if [ ${#PULL_FILES[@]} -eq 0 ]; then
    echo "[3/3] No files requested to pull. Done."
else
    echo "[3/3] Pulling ${#PULL_FILES[@]} file(s) into data/..."
    mkdir -p data
    for f in "${PULL_FILES[@]}"; do
        # Data files are in remote data/ subfolder (moved there by run_in_session1.bat)
        scp -P $PORT "${USER}@${SERVER_IP}:${WIN_DIR_FS}/data/${f}" ./data/ 2>/dev/null \
            && echo "   data/${f}" \
            || echo "   ${f} not found on remote (checked data/)"
    done
fi

echo "Done."
