#!/usr/bin/env bash
# GS_run.sh — push local commits to GitHub, sync GS, and run a script there.
#
# Usage:
#     ./GS_run.sh [--env KEY=VAL ...] <script-path> [extra args for the script...]
#
# Behavior:
#   1. Refuse to run if the local working tree has uncommitted changes
#      (commit or stash first).
#   2. git push the current branch.
#   3. SSH to GS, git fetch + reset --hard origin/<branch> in $GS_REPO.
#   4. SSH to GS, run `uv run --no-sync python <script-path> [args]`,
#      streaming output back to the local terminal.
#
# Each `--env KEY=VAL` is exported in the remote shell before the run, so
# scripts that read os.environ on GS pick it up. Multiple `--env` flags
# are allowed; they apply only to this run.
#
# Config defaults (override via env):
#   GS_HOST  199.7.140.178
#   GS_PORT  30902
#   GS_USER  GS
#   GS_REPO  SLMengineer   (path relative to GS user's home)

set -euo pipefail

REMOTE_ENV=""
while [ $# -gt 0 ] && [ "$1" = "--env" ]; do
    shift
    if [ $# -lt 1 ]; then
        echo "[GS_run] ERROR: --env requires KEY=VAL" >&2
        exit 64
    fi
    REMOTE_ENV+="set \"$1\" && "
    shift
done

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--env KEY=VAL ...] <script-path> [args...]" >&2
    exit 64
fi

SCRIPT_PATH="$1"; shift

if ! git diff --quiet HEAD --; then
    echo "[GS_run] ERROR: uncommitted changes — commit or stash first." >&2
    git status --short >&2
    exit 65
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" = "HEAD" ]; then
    echo "[GS_run] ERROR: HEAD is detached, refusing to push." >&2
    exit 66
fi

GS_HOST="${GS_HOST:-199.7.140.178}"
GS_PORT="${GS_PORT:-30902}"
GS_USER="${GS_USER:-GS}"
GS_REPO="${GS_REPO:-SLMengineer}"

echo "[1/3] git push origin ${BRANCH}..."
git push origin "${BRANCH}"

echo "[2/3] ssh ${GS_USER}@${GS_HOST}: git fetch + reset --hard origin/${BRANCH}..."
ssh -p "${GS_PORT}" "${GS_USER}@${GS_HOST}" \
    "cd ${GS_REPO} && git fetch origin && git reset --hard origin/${BRANCH}"

echo "[3/3] ssh ${GS_USER}@${GS_HOST}: ${REMOTE_ENV}uv run --no-sync python ${SCRIPT_PATH} $*..."
ssh -p "${GS_PORT}" "${GS_USER}@${GS_HOST}" \
    "cd ${GS_REPO} && ${REMOTE_ENV}uv run --no-sync python ${SCRIPT_PATH} $*"
