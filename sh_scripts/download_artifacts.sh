#!/usr/bin/env bash
# =============================================================================
# Download the whole artifacts/ folder from a RunPod pod (or any SSH server) to this checkout.
# Run it on YOUR PC (Git Bash on Windows, or Linux/macOS):
#
#   bash sh_scripts/download_artifacts.sh root@<IP> <PORT>
#   SSH_KEY=~/.ssh/id_rsa bash sh_scripts/download_artifacts.sh root@<IP> <PORT>
#   INCLUDE_CHECKPOINTS=1 bash sh_scripts/download_artifacts.sh root@<IP> <PORT>
#
# <IP> and <PORT> are shown in RunPod -> Connect -> "SSH over exposed TCP".
#
# Env:
#   REMOTE_DIR          [/workspace/DIVERGE]  repo location on the pod
#   SSH_KEY             [~/.ssh/id_ed25519]   private key whose .pub is in RunPod settings
#   INCLUDE_CHECKPOINTS [0]                   1 = also download HF Trainer checkpoint-* dirs (large)
#   KEEP_TAR            [1]                   keep the downloaded tar.gz in exports/
#
# The pod streams artifacts/ as a tar.gz over SSH into exports/ (nothing is written on the pod),
# then restore_outputs.sh merges it in: adapters you already have locally are kept (OVERWRITE=1 to replace).
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

HOST="${1:?usage: download_artifacts.sh root@<IP> <PORT>}"
PORT="${2:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/DIVERGE}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-0}"
KEEP_TAR="${KEEP_TAR:-1}"

SSH=(ssh -p "$PORT" -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new)
[[ -f "$SSH_KEY" ]] && SSH+=(-i "$SSH_KEY")

EXCLUDE=""
[[ "$INCLUDE_CHECKPOINTS" == "1" ]] || EXCLUDE="--exclude=checkpoint-*"

log "Remote size of $REMOTE_DIR/artifacts:"
"${SSH[@]}" "$HOST" "du -sh '$REMOTE_DIR/artifacts'"

mkdir -p exports
OUT="exports/artifacts_$(date '+%Y%m%d-%H%M%S').tar.gz"
PART="$OUT.part"
log "Downloading -> $OUT"
"${SSH[@]}" "$HOST" "cd '$REMOTE_DIR' && tar $EXCLUDE -czf - artifacts" > "$PART"
mv "$PART" "$OUT"    # only a complete download gets the final name
ls -lh "$OUT"

log "Checking archive"
tar -tzf "$OUT" > /dev/null

bash "$REPO_ROOT/sh_scripts/restore_outputs.sh" "$REPO_ROOT/$OUT"

if [[ "$KEEP_TAR" != "1" ]]; then rm -f "$OUT"; fi
log "Download done."
