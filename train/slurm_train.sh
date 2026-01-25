#!/bin/bash
#SBATCH -J agentflow_train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p long
#SBATCH -t 24:00:00
#SBATCH -o logs/agentflow_%j.out
#SBATCH -e logs/agentflow_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -e


mkdir -p /home/artem.shelmanov/vlad/AgentFlow/logs

# Increase file descriptor limit for Ray (try multiple approaches)
# First, try to set soft limit to hard limit
ulimit -Sn $(ulimit -Hn) 2>/dev/null || true
# Then try explicit values
ulimit -n 65536 2>/dev/null || ulimit -n 16384 2>/dev/null || ulimit -n 8192 2>/dev/null || ulimit -n 4096 2>/dev/null || true

# Limit thread spawning
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NUMEXPR_MAX_THREADS=1
export RAY_DEDUP_LOGS=0
export RAY_num_server_call_thread=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=0
# Limit Ray's resource usage to work around file descriptor limits
export RAY_max_calls_in_flight_per_worker=1
export RAY_object_store_memory=1000000000  # 1GB object store
export RAY_num_workers=2  # Limit number of workers
export RAY_worker_register_timeout_seconds=120

export CUDA_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES

# Load API keys
export OPENAI_API_KEY="${OPENAI_API_KEY}"

# Use system python3 with user packages
export PATH="/home/artem.shelmanov/.local/bin:$PATH"

# Install flash-attn from prebuilt wheel (torch 2.6 + cu124 + cp310)
echo "Checking PyTorch version..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

echo "Installing flash-attn from prebuilt wheel..."
pip uninstall flash-attn -y 2>/dev/null || true

# Try to install flash-attn (optional - config uses eager attention as fallback)
WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl"
echo "Installing from: $WHEEL_URL"
if pip install "$WHEEL_URL" 2>&1; then
    python3 -c "from flash_attn.flash_attn_interface import flash_attn_func; print('flash-attn installed and working!')" || echo "flash-attn verification failed, using eager attention"
else
    echo "WARNING: Could not install flash-attn, continuing with eager attention"
fi

# Unbuffered Python output for SLURM logs
export PYTHONUNBUFFERED=1

cd /home/artem.shelmanov/vlad/AgentFlow

# Clean up Ray cache
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray /tmp/slurm-$(whoami)-* 2>/dev/null || true

# Start rollout service in background
python3 train/rollout.py &
ROLLOUT_PID=$!

# Run training script
python3 train/train_agent.py
TRAIN_EXIT=$?

kill $ROLLOUT_PID 2>/dev/null || true
ray stop --force 2>/dev/null || true
exit $TRAIN_EXIT
