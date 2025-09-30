#!/bin/bash

# ===========================================================================
# Script: serve_multiple_models.sh
# Description:
#   Launch multiple vLLM model servers in parallel using tmux.
#   Each model uses:
#     - 2 GPUs (via CUDA_VISIBLE_DEVICES)
#     - tensor-parallel-size=2
#     - Port starting from 8000
#   Models are predefined in the script (no user input needed).
#
# Requirements:
#   - vLLM: pip install vllm
#   - tmux: sudo apt install tmux
# ===========================================================================

# -------------------------------
# 🔧 CONFIGURATION SECTION
# -------------------------------

# Define the list of models to serve (edit this list directly) 
# DO NOT EXCEED THE NUMBER OF FOLLOWING GPU GROUPS

models=(
    # "checkpoints/.../actor/huggingface"
    "IPF/AgentFlow-7B"
    # "TIGER-Lab/General-Reasoner-Qwen2.5-7B"
    # "hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo"
    # "Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
    # "Elliott/LUFFY-Qwen-Instruct-7B"
    # "google/gemma-2-9b"
    # "mistralai/Mistral-7B-v0.1"
)

# Available GPU groups (each group uses 2 GPUs)
gpu_groups=(
    "0"
    # "2"
    # "0,1,2,3"
    # "2,3"
    # "4,5,6,7"
    # "6,7"
)
num_gpu_groups=${#gpu_groups[@]}

# Starting port for the first model
start_port=8000
# start_port=9000

# Tmux session name
tmux_session="vllm_models"

# -------------------------------
# 🚀 SCRIPT EXECUTION
# -------------------------------

# Check if any models are defined
if [ ${#models[@]} -eq 0 ]; then
    echo "No models defined in script. Exiting."
    exit 1
fi

echo "Starting ${#models[@]} models with 2-GPU Tensor Parallelism..."

PROJECT_DIR="/home/user/my-vllm-project"
# TODO: delete
PROJECT_DIR="/home/ubuntu/jianwen-us-midwest-1/panlu/ipf-new/AgentFlow"
VENV_ACTIVATE="source .venv/bin/activate"

# Create a new detached tmux session
tmux new-session -d -s "$tmux_session"

port=$start_port
gpu_group_index=0

for model in "${models[@]}"; do
    # Get current GPU group
    gpu_list=${gpu_groups[$gpu_group_index]}
    
    # 🔢 Automatically calculate tensor parallel size
    tp_size=$(echo $gpu_list | awk -F',' '{print NF}')
    
    # Set CUDA_VISIBLE_DEVICES (only for environment echo, not needed in tmux if set later)
    export CUDA_VISIBLE_DEVICES=$gpu_list

    echo "Launching model: $model"
    echo "  Port: $port"
    echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
    echo "  Tensor Parallel Size: $tp_size"

    # Build the command with cd + activate + vLLM serve
    CMD_START="
        cd \"$PROJECT_DIR\";
        $VENV_ACTIVATE;
        export CUDA_VISIBLE_DEVICES=$gpu_list;
        echo '--- Starting $model on port $port with TP=$tp_size ---';
        echo 'CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES';
        echo 'Current virtual env: \$(python -c \"import sys; print(sys.prefix)\")';
        vllm serve \"$model\" \
            --host 0.0.0.0 \
            --port $port \
            --tensor-parallel-size $tp_size \
    "

    if [ "$gpu_group_index" -eq 0 ]; then
        # First model in the first pane
        tmux send-keys -t "${tmux_session}:0" "$CMD_START" C-m
    else
        # Split window and launch in new pane
        tmux split-window -h -t "$tmux_session:0"
        sleep 0.2  # Avoid race condition
        tmux send-keys -t "${tmux_session}:0" "$CMD_START" C-m
    fi

    # Increment port and rotate GPU group
    ((port++))
    ((gpu_group_index = (gpu_group_index + 1) % num_gpu_groups))
done

# Optimize layout
tmux select-layout -t "$tmux_session:0" tiled

# Print instructions
echo ""
echo "✅ All models launched in tmux session: '$tmux_session'"
echo "💡 View logs:   tmux attach-session -t $tmux_session"
echo "💡 Detach:      Ctrl+B, then D"
echo "💡 Kill session: tmux kill-session -t $tmux_session"



# export CUDA_VISIBLE_DEVICES=4
# echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
# echo Current virtual env: $(python -c "import sys; print(sys.prefix)")
# vllm serve "checkpoints/tmp/.../huggingface" \
#     --host 0.0.0.0 \
#     --port 8004 \
#     --tensor-parallel-size 1

