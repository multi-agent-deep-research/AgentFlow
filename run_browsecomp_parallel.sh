#!/bin/bash
#
# Run BrowseComp evaluation with parallel workers sharing a single embedding server.
#
# Architecture:
#   [Embedding Server (vLLM, 1 GPU)] <-- HTTP API
#       ^   ^   ^   ^
#       |   |   |   |
#   [Worker 1] [Worker 2] [Worker 3] [Worker 4]  (no GPU needed per worker)
#
# Each worker runs the full eval script. Workers skip already-completed queries
# (via output_file.exists() check), so they naturally load-balance.
#
# Usage:
#   # 1. Start embedding server first (in another terminal or background):
#   CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-8B \
#       --port 8002 --task embed \
#       --override-pooler-config '{"pooling_type": "LAST", "normalize": true}' \
#       --gpu-memory-utilization 0.3
#
#   # 2. Run parallel eval:
#   bash run_browsecomp_parallel.sh --workers 4 \
#       --service hosted_vllm --model Qwen/Qwen3.5-9B \
#       --max-steps 20 --output-dir runs/agentflow_parallel
#
# Requirements:
#   - EMBEDDING_API_BASE must be set (default: http://localhost:8002/v1)
#   - BROWSECOMP_INDEX_PATH and BROWSECOMP_INDEX_TYPE=faiss must be set
#   - vLLM planner server on port 8000 (for hybrid mode)
#   - vLLM executor server on port 8001 (for hosted_vllm service)

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

# Default embedding API
export EMBEDDING_API_BASE="${EMBEDDING_API_BASE:-http://localhost:8002/v1}"

echo "========================================"
echo "BrowseComp Parallel Evaluation"
echo "========================================"
echo "Workers:        $WORKERS"
echo "Embedding API:  $EMBEDDING_API_BASE"
echo "Eval args:      ${EVAL_ARGS[*]}"
echo "========================================"

# Verify embedding server is up
if ! curl -s "$EMBEDDING_API_BASE/models" > /dev/null 2>&1; then
    echo "ERROR: Embedding server not reachable at $EMBEDDING_API_BASE"
    echo ""
    echo "Start it with:"
    echo "  CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-8B \\"
    echo "      --port 8002 --task embed \\"
    echo "      --override-pooler-config '{\"pooling_type\": \"LAST\", \"normalize\": true}' \\"
    echo "      --gpu-memory-utilization 0.3"
    exit 1
fi
echo "Embedding server OK"

# Launch workers (no GPU needed per worker since embeddings are remote)
PIDS=()
for i in $(seq 1 "$WORKERS"); do
    echo "Starting worker $i/$WORKERS..."
    python run_browsecomp_eval.py "${EVAL_ARGS[@]}" > /dev/null 2>&1 &
    PIDS+=($!)
    sleep 2  # stagger to avoid race on wandb init
done

echo ""
echo "All $WORKERS workers started (PIDs: ${PIDS[*]})"
echo ""

# Wait for all workers
echo "Waiting for all workers to finish..."
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo "All workers finished!"
