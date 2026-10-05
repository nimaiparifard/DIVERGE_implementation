#!/usr/bin/env bash
# =============================================================================
# Run on YOUR machine (Git Bash): pack the code (tracked + untracked, minus data, models,
# outputs) into exports/diverge_code.tar.gz - use this when you have not pushed to GitHub.
#
#   bash sh_scripts/pack_code.sh
#   # upload to Drive / RunPod, then:  mkdir DIVERGE && tar -xzf diverge_code.tar.gz -C DIVERGE
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

mkdir -p exports
# git knows tracked + untracked-but-not-ignored files; drop heavy output folders on top of .gitignore
git ls-files -z --cached --others --exclude-standard |
    grep -zvE '^(artifacts|outputs|outputs_eff|exports|logs|results|datasets|local_models)/' |
    grep -zvE '\.(pt|pth|safetensors|bin|tar\.gz|zip)$' |
    tar --null -T - -czf exports/diverge_code.tar.gz
ls -lh exports/diverge_code.tar.gz
log "Files: $(tar -tzf exports/diverge_code.tar.gz | wc -l)"
