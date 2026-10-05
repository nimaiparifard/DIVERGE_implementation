#!/usr/bin/env bash
# Apply a seed snapshot exported on another machine (see scripts/import_seeds.py)
#   bash sh_scripts/import_seeds.sh                                    # snapshot already at ./seed_snapshot
#   bash sh_scripts/import_seeds.sh --snapshot /workspace/seed_snapshot
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" scripts/import_seeds.py "$@"
