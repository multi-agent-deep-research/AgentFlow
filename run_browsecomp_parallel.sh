#!/bin/bash
#
# Run BrowseComp evaluation with parallel workers.
# Each worker gets a non-overlapping partition of queries via --worker-id/--num-workers.
#
# Prerequisites:
#   - Embedding server on port 8002 (EMBEDDING_API_BASE)
#   - LLM server on port 8001 or 8003 (HOSTED_VLLM_API_BASE)
#   - BROWSECOMP_INDEX_PATH and BROWSECOMP_INDEX_TYPE set
#
# Usage:
#   bash run_browsecomp_parallel.sh --workers 4 \
#       --unified --service hosted_vllm --model Qwen/Qwen3.5-9B \
#       --max-steps 20 --output-dir runs/setup_b_parallel

set -e

WORKERS=4
EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            EVAL_ARGS+=("$1")
            shift
            ;;
    esac
done

export EMBEDDING_API_BASE="${EMBEDDING_API_BASE:-http://localhost:8002/v1}"

echo "========================================"
echo "BrowseComp Parallel Evaluation"
echo "========================================"
echo "Workers:        $WORKERS"
echo "Embedding API:  $EMBEDDING_API_BASE"
echo "Eval args:      ${EVAL_ARGS[*]}"
echo "========================================"

# Verify embedding server
if ! curl -s "$EMBEDDING_API_BASE/models" > /dev/null 2>&1; then
    echo "ERROR: Embedding server not reachable at $EMBEDDING_API_BASE"
    exit 1
fi
echo "Embedding server OK"

# Launch workers with non-overlapping query partitions
PIDS=()
for i in $(seq 0 $((WORKERS - 1))); do
    echo "Starting worker $((i+1))/$WORKERS (worker_id=$i)..."
    python run_browsecomp_eval.py "${EVAL_ARGS[@]}" \
        --worker-id "$i" --num-workers "$WORKERS" > /dev/null 2>&1 &
    PIDS+=($!)
    sleep 2
done

echo ""
echo "All $WORKERS workers started (PIDs: ${PIDS[*]})"
echo "Each worker processes ~$((830 / WORKERS)) queries (non-overlapping partitions)"
echo ""

# Wait for all workers
echo "Waiting for all workers to finish..."
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo "All workers finished!"
