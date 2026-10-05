#!/usr/bin/env bash
# Job pool: run several GPU jobs at the same time on ONE GPU (e.g. a rented 4090).
# Sourced by sh_scripts/run_*_parallel.sh after _common.sh - not meant to be run directly.
#
# A new job is launched only when BOTH hold:
#   - fewer than MAX_JOBS jobs are running
#   - the GPU has at least MIN_FREE_GB of free VRAM (checked with nvidia-smi)
# so the pool fills the card without OOM-ing it. A job that still dies with CUDA OOM
# (VRAM grows after the model is loaded) is retried up to OOM_RETRIES times.
#
# Env (defaults are set by the calling script):
#   MAX_JOBS      max concurrent jobs
#   MIN_FREE_GB   free VRAM needed before launching the next job
#   STAGGER_SEC   pause after every launch so the job allocates its VRAM before the next check
#   OOM_RETRIES   retries for a job that failed with CUDA out-of-memory
#   GPU_ID [0]    which GPU to watch / run on
#
# Caller fills these arrays (same index = same job), sets HISTORY_FILE / FAILED_FILE, then calls pool_run:
#   JOB_NAMES  human-readable run id
#   JOB_META   CSV fields written into HISTORY_FILE between status and exit_code
#   JOB_CMDS   command line (quote it with printf %q)
#   JOB_LOGS   log file of the job
#   JOB_CLEAN  command run after a successful job ("" = nothing)

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
LOCK_FILE="$STATUS_DIR/.parallel.lock"
JOB_NAMES=(); JOB_META=(); JOB_CMDS=(); JOB_LOGS=(); JOB_CLEAN=()
PIDS=()

# locked FN ARGS...   run a shell function while holding the pool lock (history CSV / tally writes)
locked() {
    if command -v flock >/dev/null 2>&1; then
        { flock 9; "$@"; } 9>"$LOCK_FILE"
    else
        "$@"
    fi
}

gpu_query() {  # gpu_query memory.free|memory.total|utilization.gpu  -> number (MB / %)
    nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | head -n1 | tr -d ' '
}
gpu_free_mb() { local v; v="$(gpu_query memory.free)"; echo "${v:-999999}"; }

# Split the CPU cores between the jobs (tokenizers / OpenMP oversubscribe badly otherwise)
_threads_per_job() {
    local cores; cores="$(nproc 2>/dev/null || echo 8)"
    local t=$(( cores / MAX_JOBS )); (( t < 1 )) && t=1; echo "$t"
}

_pool_record() {  # _pool_record IDX RC START END
    local i="$1" rc="$2" t0="$3" t1="$4"
    if [[ "$rc" -eq 0 ]]; then
        echo "$t1,OK,${JOB_META[$i]},0" >> "$HISTORY_FILE"
        echo "OK" >> "$POOL_TALLY"
        # no sync_outputs here: other jobs may be half-way through writing their files and
        # merge_tree never overwrites -> the caller syncs once after pool_run
    else
        echo "$t1,FAILED,${JOB_META[$i]},$rc" >> "$HISTORY_FILE"
        echo "$t1 FAILED ${JOB_NAMES[$i]} exit=$rc log=${JOB_LOGS[$i]}" >> "$FAILED_FILE"
        echo "FAILED" >> "$POOL_TALLY"
    fi
}

_pool_job() {  # runs in the background, one per job
    local i="$1" attempt=0 rc t0 logf="${JOB_LOGS[$1]}"
    export OMP_NUM_THREADS MKL_NUM_THREADS TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="$(_threads_per_job)"; MKL_NUM_THREADS="$OMP_NUM_THREADS"
    t0="$(ts)"
    while :; do
        ( eval "${JOB_CMDS[$i]}" ) > "$logf" 2>&1
        rc=$?
        if [[ $rc -ne 0 && $attempt -lt $OOM_RETRIES ]] && grep -qiE "out of memory|OutOfMemoryError" "$logf"; then
            attempt=$((attempt + 1))
            mv -f "$logf" "$logf.oom$attempt"
            log "OOM   ${JOB_NAMES[$i]}  -> retry $attempt/$OOM_RETRIES when VRAM frees up"
            sleep ${OOM_SLEEP:-$(( 30 + RANDOM % 60 ))}
            while (( $(gpu_free_mb) < MIN_FREE_GB * 1024 )); do sleep 10; done
            continue
        fi
        break
    done
    if [[ $rc -eq 0 ]]; then
        log "DONE  ${JOB_NAMES[$i]}"
        [[ -n "${JOB_CLEAN[$i]}" ]] && eval "${JOB_CLEAN[$i]}"
    else
        log "FAIL  ${JOB_NAMES[$i]}  exit=$rc  (log: $logf)"
    fi
    locked _pool_record "$i" "$rc" "$t0" "$(ts)"
    return $rc
}

_pool_reap() {  # drop finished PIDs
    local p alive=()
    for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    PIDS=(${alive[@]+"${alive[@]}"})
}

_pool_status() {  # snapshot of what is running, for `cat results/run_status/PARALLEL_*.txt`
    {
        echo "updated=$(ts) host=$(hostname)"
        echo "running=${#PIDS[@]} max_jobs=$MAX_JOBS launched=$1/$2"
        echo "gpu_free_mb=$(gpu_free_mb) gpu_total_mb=$(gpu_query memory.total) gpu_util=$(gpu_query utilization.gpu)%"
    } > "$POOL_STATUS_FILE"
}

# pool_run LABEL   -> runs every queued job, returns when all are finished
pool_run() {
    local label="$1" n=${#JOB_CMDS[@]} i
    POOL_TALLY="$STATUS_DIR/.pool_${label}_$$.tally"; : > "$POOL_TALLY"
    POOL_STATUS_FILE="$STATUS_DIR/PARALLEL_${label^^}_STATUS.txt"
    log "pool: $n job(s)  MAX_JOBS=$MAX_JOBS  MIN_FREE_GB=$MIN_FREE_GB  STAGGER_SEC=$STAGGER_SEC  OOM_RETRIES=$OOM_RETRIES  threads/job=$(_threads_per_job)"
    for ((i = 0; i < n; i++)); do
        while :; do
            _pool_reap
            _pool_status "$i" "$n"
            if (( ${#PIDS[@]} == 0 )); then break; fi
            if (( ${#PIDS[@]} < MAX_JOBS && $(gpu_free_mb) >= MIN_FREE_GB * 1024 )); then break; fi
            sleep 5
        done
        _pool_job "$i" &
        PIDS+=("$!")
        log "START [$((i + 1))/$n] ${JOB_NAMES[$i]}   (running=${#PIDS[@]}, free VRAM=$(gpu_free_mb)MB, log: ${JOB_LOGS[$i]})"
        (( i < n - 1 )) && sleep "$STAGGER_SEC"
    done
    wait
    _pool_reap; _pool_status "$n" "$n"
    POOL_OK=$(grep -c '^OK' "$POOL_TALLY" || true)
    POOL_FAIL=$(grep -c '^FAILED' "$POOL_TALLY" || true)
    rm -f "$POOL_TALLY"
}
