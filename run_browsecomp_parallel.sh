#!/bin/bash
#
# Run BrowseComp evaluation with parallel workers.
# Each worker gets a non-overlapping partition of queries via --worker-id/--num-workers.
# All workers share one run directory with per-worker subdirectories.
#
# Output structure:
#   runs/output_dir/20260324_215443/worker_0/
#   runs/output_dir/20260324_215443/worker_1/
#   runs/output_dir/20260324_215443/worker_2/
#   runs/output_dir/20260324_215443/worker_3/
#
# Prerequisites:
#   - Embedding server on port 8002 (EMBEDDING_API_BASE)
#   - LLM server (HOSTED_VLLM_API_BASE)
#   - BROWSECOMP_INDEX_PATH and BROWSECOMP_INDEX_TYPE set
#
# Usage:
#   bash run_browsecomp_parallel.sh --workers 4 \
#       --unified --service hosted_vllm --model Qwen/Qwen3.5-9B \
#       --max-steps 20 --output-dir runs/setup_b_parallel
#
#   # In background:
#   nohup bash run_browsecomp_parallel.sh --workers 4 \
#       --unified --service hosted_vllm --model Qwen/Qwen3.5-9B \
#       --max-steps 20 --output-dir runs/setup_b_parallel > /dev/null 2>&1 &

set -e

WORKERS=4
OUTPUT_DIR=""
EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            EVAL_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            EVAL_ARGS+=("$1")
            shift
            ;;
    esac
done

export EMBEDDING_API_BASE="${EMBEDDING_API_BASE:-http://localhost:8002/v1}"

# Create shared timestamp directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="runs/agentflow"
fi
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "========================================"
echo "BrowseComp Parallel Evaluation"
echo "========================================"
echo "Workers:        $WORKERS"
echo "Embedding API:  $EMBEDDING_API_BASE"
echo "Run directory:  $RUN_DIR"
echo "Eval args:      ${EVAL_ARGS[*]}"
echo "========================================"

# Verify embedding server
if ! curl -s "$EMBEDDING_API_BASE/models" > /dev/null 2>&1; then
    echo "ERROR: Embedding server not reachable at $EMBEDDING_API_BASE"
    exit 1
fi
echo "Embedding server OK"

# Remove --output-dir from EVAL_ARGS since we use --run-dir instead
FILTERED_ARGS=()
SKIP_NEXT=false
for arg in "${EVAL_ARGS[@]}"; do
    if $SKIP_NEXT; then
        SKIP_NEXT=false
        continue
    fi
    if [ "$arg" = "--output-dir" ]; then
        SKIP_NEXT=true
        continue
    fi
    FILTERED_ARGS+=("$arg")
done

# Launch workers with non-overlapping query partitions
PIDS=()
for i in $(seq 0 $((WORKERS - 1))); do
    WORKER_DIR="${RUN_DIR}/worker_${i}"
    mkdir -p "$WORKER_DIR"
    echo "Starting worker $((i+1))/$WORKERS (worker_id=$i) -> $WORKER_DIR"
    python run_browsecomp_eval.py "${FILTERED_ARGS[@]}" \
        --worker-id "$i" --num-workers "$WORKERS" \
        --run-dir "$WORKER_DIR" > /dev/null 2>&1 &
    PIDS+=($!)
    sleep 2
done

echo ""
echo "All $WORKERS workers started (PIDs: ${PIDS[*]})"
echo "Each worker processes ~$((830 / WORKERS)) queries (non-overlapping partitions)"
echo ""
echo "Check progress:"
echo "  ls $RUN_DIR/worker_*/*.json | wc -l"
echo "  grep 'Running judge accuracy' $RUN_DIR/worker_*/eval.log | tail -$WORKERS"
echo ""

# Wait for all workers
echo "Waiting for all workers to finish..."
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo "All workers finished!"
echo "Results in: $RUN_DIR"
echo "Total results: $(ls $RUN_DIR/worker_*/*.json 2>/dev/null | grep -v summary | grep -v judge | wc -l)"
