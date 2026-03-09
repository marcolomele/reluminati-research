#!/bin/bash
# monitor_jobs.sh — watches SLURM jobs, detects errors, logs findings
# Designed to run on the login node via: nohup bash monitor_jobs.sh > monitor.log 2>&1 &
#
# Usage:
#   nohup bash monitor_jobs.sh > monitor.log 2>&1 &
#   tail -f monitor.log   (to watch from another session)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/monitor.log"
USER_ID=3164542
POLL_INTERVAL=60   # seconds between squeue polls

# Job IDs to watch (filled at launch time)
declare -A JOB_LABELS=(
    [465388]="EXP-C-box/cot (resume from pair 371)"
    [465389]="EXP-F-mf1/3/5/10 (resume from pair 126)"
)

# Error patterns that indicate a broken run (grep -E compatible)
# NOTE: "[ERROR] Pair N failed: Image not found" is benign — handled in try/except, job continues.
# Only flag truly fatal errors (Traceback, OOM, CUDA crash, etc.)
ERROR_PATTERNS=(
    "Traceback \(most recent"
    "CUDA error|CUDA out of memory|out of memory|OOM"
    "Sequence length.*exceeds"
    "Connection refused|ConnectionRefused"
    "^Killed"
    "DUE TO TIME LIMIT"
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

ts() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

job_state() {
    squeue -j "$1" -h -o "%T" 2>/dev/null || sacct -j "$1" --format=State --noheader 2>/dev/null | head -1 | xargs
}

# ─── Error detection ──────────────────────────────────────────────────────────

check_err_file() {
    local errfile="$1"
    local jobid="$2"
    [[ -f "$errfile" ]] || return 0

    local found=""
    for pat in "${ERROR_PATTERNS[@]}"; do
        if grep -qE "$pat" "$errfile" 2>/dev/null; then
            found="$pat"
            break
        fi
    done

    if [[ -n "$found" ]]; then
        log "!! ERROR detected in $errfile (pattern: $found)"
        log "   Last 30 lines of .err:"
        tail -30 "$errfile" | while IFS= read -r line; do log "   ERR> $line"; done
        return 1
    fi
    return 0
}

check_out_file() {
    local outfile="$1"
    [[ -f "$outfile" ]] || return 0
    # Log last progress line (pair X/Y)
    local progress
    progress=$(grep -oE "Pair [0-9]+/[0-9]+" "$outfile" 2>/dev/null | tail -1)
    [[ -n "$progress" ]] && log "   Progress: $progress"
}

# ─── Per-job monitoring loop ──────────────────────────────────────────────────

monitor_job() {
    local jobid="$1"
    local label="${JOB_LABELS[$jobid]:-unknown}"
    local errfile="$SCRIPT_DIR/smoke_${jobid}.err"
    local outfile="$SCRIPT_DIR/smoke_${jobid}.out"
    local prev_state=""
    local error_seen=0

    log "=== Monitoring job $jobid ($label) ==="

    while true; do
        local state
        state=$(job_state "$jobid")
        state="${state%%.*}"  # strip trailing dots

        if [[ "$state" != "$prev_state" ]]; then
            log "Job $jobid ($label): $prev_state -> $state"
            prev_state="$state"
        fi

        case "$state" in
            PENDING)
                sleep "$POLL_INTERVAL"
                continue
                ;;
            RUNNING)
                check_out_file "$outfile"
                if ! check_err_file "$errfile" "$jobid"; then
                    error_seen=1
                    log "!! Cancelling job $jobid due to detected error"
                    scancel "$jobid"
                    log "   scancel sent. Manual intervention needed."
                    log "   Error file: $errfile"
                    log "   See monitor.log for context."
                    return 1
                fi
                sleep "$POLL_INTERVAL"
                ;;
            COMPLETED)
                log "Job $jobid ($label): COMPLETED OK"
                # Final check
                check_out_file "$outfile"
                check_err_file "$errfile" "$jobid" && log "No errors in final .err"
                # Show summary stats if in .out
                local summary
                summary=$(grep -A 20 "EXPERIMENT SUMMARY" "$outfile" 2>/dev/null | head -25)
                if [[ -n "$summary" ]]; then
                    log "=== SUMMARY for job $jobid ==="
                    echo "$summary" | while IFS= read -r line; do log "   $line"; done
                fi
                return 0
                ;;
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                log "!! Job $jobid ($label): terminal state $state"
                check_out_file "$outfile"
                check_err_file "$errfile" "$jobid" || true
                return 1
                ;;
            "")
                # Job no longer in squeue or sacct — treat as done
                log "Job $jobid ($label): no longer found in queue (assumed completed or expired)"
                [[ -f "$outfile" ]] && check_out_file "$outfile"
                [[ -f "$errfile" ]] && check_err_file "$errfile" "$jobid" || true
                return 0
                ;;
            *)
                log "Job $jobid ($label): unknown state '$state', retrying..."
                sleep "$POLL_INTERVAL"
                ;;
        esac
    done
}

# ─── Main ─────────────────────────────────────────────────────────────────────

log "========================================="
log " monitor_jobs.sh started (PID $$)"
log " Watching jobs: ${!JOB_LABELS[*]}"
log " Poll interval: ${POLL_INTERVAL}s"
log " Log: $LOG"
log "========================================="

# Monitor all jobs in parallel (background subshells)
pids=()
for jobid in "${!JOB_LABELS[@]}"; do
    monitor_job "$jobid" &
    pids+=($!)
done

# Wait for all monitors to finish and collect exit codes
all_ok=0
for pid in "${pids[@]}"; do
    wait "$pid" || all_ok=1
done

log "========================================="
if [[ $all_ok -eq 0 ]]; then
    log " All jobs completed successfully."
else
    log " One or more jobs ended with errors. Check monitor.log."
fi
log " monitor_jobs.sh done."
log "========================================="
