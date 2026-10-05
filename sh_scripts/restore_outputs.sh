#!/usr/bin/env bash
# =============================================================================
# Merge outputs produced on another machine into this checkout.
# Works on Linux AND on Windows (Git Bash):
#
#   bash sh_scripts/restore_outputs.sh exports/diverge_outputs_<host>_<stamp>.tar.gz
#   bash sh_scripts/restore_outputs.sh /content/drive/MyDrive/DIVERGE_sync      # a SYNC_DIR folder
#
# Safe to run repeatedly:
#   - adapters / caches / reports that already exist here are kept (OVERWRITE=1 to replace)
#   - CSVs (run_history, efficiency, diverge_results, ...) get the new rows appended, duplicates dropped
# Afterwards, the run scripts with SKIP_EXISTING=1 (default) continue with only what is missing.
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SRC="${1:?usage: restore_outputs.sh <export.tar.gz | sync dir>}"
case "$SRC" in /*|[A-Za-z]:*) ;; *) SRC="$OLDPWD/$SRC" ;; esac   # relative to where you called it
[[ -e "$SRC" ]] || SRC="${1}"                                    # ...or relative to the repo root

if [[ -f "$SRC" ]]; then
    TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
    log "Extracting $SRC"
    tar -xzf "$SRC" -C "$TMP"
    SRC="$TMP"
fi

for p in artifacts results; do
    if [[ -d "$SRC/$p" ]]; then
        log "Merging $p/"
        merge_tree "$SRC/$p" "$REPO_ROOT/$p"
    fi
done
log "Restore done."
bash "$REPO_ROOT/sh_scripts/list_runs.sh"
