#!/bin/bash

# ===========================================================================
# Script: serve_vllm_qwen.sh
# Description:
#   Launch *** using vLLM in a tmux window
#   - Uses GPU ***
#   - tensor-parallel-size=***
#   - Port *** (different from planner model on port 8000) !!!!!
#   !!!! it is important to check that the served during training model
#   !!!! has differenty port than ***
#   !!!! this info is printed in logs; see `grep -r "SERVED AT "` 
# 
# 
# === Model launched in tmux session: 'vllm_qwen'
# === View logs:   tmux attach-session -t vllm_qwen
# === Detach:      Ctrl+B, then D
# === Kill session: tmux kill-session -t vllm_qwen
# ===========================================================================

MODEL="Qwen/Qwen2.5-7B-Instruct"
GPU="2,3"
PORT=8000
TMUX_SESSION="vllm_qwen"
TP=2
UTILIZATION=0.75


VENV_ACTIVATE="source .venv/bin/activate"

echo "Launching model: $MODEL"
echo "  Port: $PORT"
echo "  GPU: $GPU"
echo "  Tensor Parallel Size: $TP"

# Create tmux session and run vLLM
tmux new-session -d -s "$TMUX_SESSION"

CMD_START="
    $VENV_ACTIVATE;
    export CUDA_VISIBLE_DEVICES=$GPU;
    echo '--- Starting $MODEL on port $PORT with TP=$TP ---';
    echo 'CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES';
    echo 'Current virtual env: \$(python -c \"import sys; print(sys.prefix)\")';
    vllm serve \"$MODEL\" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP \
        --gpu-memory-utilization $UTILIZATION
"

tmux send-keys -t "${TMUX_SESSION}:0" "$CMD_START" C-m

echo ""
echo "=== Model launched in tmux session: '$TMUX_SESSION'"
echo "=== View logs:   tmux attach-session -t $TMUX_SESSION"
echo "=== Detach:      Ctrl+B, then D"
echo "=== Kill session: tmux kill-session -t $TMUX_SESSION"