PLEASE CHANGE THE ARXIV, REPO, AND OTEHR URL in `agentflow/pyproject.toml`

# AgentFlow: In-The-Flow Agentic System Optimization for Effective Planning and Tool Use.

## Installation
please revise the project path in `setup.sh`, if you are not currently at the root of the project, and then run:

```bash
bash setup.sh
source .venv/bin/activate
# (Optional) Install `parallel` for running benchmark experiments in parallel:
sudo apt-get update
sudo apt-get install parallel
```

Make .env file, and set `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CX`, etc. For example:
```text
# The content of the .env file

# Used for LLM-powered modules and tools
OPENAI_API_KEY=<your-api-key-here> # If you want to use OpenAI LLM
DASHSCOPE_API_KEY=<your-api-key-here> # If you want to use Gemini LLM

# Used for the Google Search tool
GOOGLE_API_KEY=<your-api-key-here>
GOOGLE_CX=<your-cx-here>

# Used for the more custom llm-engine (Optional)
TOGETHER_API_KEY=<your-api-key-here> # If you want to use TogetherAI LLM
DEEPSEEK_API_KEY=<your-api-key-here> # If you want to use DeepSeek LLM
XAI_API_KEY=<your-api-key-here> # If you want to use Grok LLM
ANTHROPIC_API_KEY=<your-api-key-here> # If you want to use Anthropic LLM
```

## Quick Start
### Dataset Preparation
Please run the following commands to fetch the data:
```bash
# train data
python data/get_train_data.py
# validation data
python data/aime24_data.py
```

The data dir should be:
```
data/
├── train/
│   └── combined_train.parquet (182,190 samples)
├── val/
│   └── aime24.parquet (30 samples)
├── aime24_data.py
└── get_train_data.py
```
### Train
**Start training with tmux:**
```bash
# Create tmux session and start agentflow service (Window 0)
tmux new-session -s agentflow
bash train/serve_with_logs.sh

# Create new window (Ctrl+B then C) and start training (Window 1)
bash train/train_with_logs.sh
```
**Configuration:**
All training hyperparameters are in `train/config.yaml` (model settings, tools, PPO parameters, resources, etc.)











## Test your env before going on

vplease run the following command to test all tools:

```bash
cd agentflow/agentflow
bash ./tools/test_all_tools.sh
```

A `test.log` will be saved in each tool's file. 

Success example: 
```text
Testing all tools
Tools:
  - base_generator
  - google_search
  - python_coder
  - web_search
  - wikipedia_search

Running tests in parallel...
Testing base_generator...
✅ base_generator passed
Testing google_search...
✅ google_search passed
Testing python_coder...
✅ python_coder passed
Testing wikipedia_search...
✅ wikipedia_search passed
Testing web_search...
✅ web_search passed

✅ All tests passed
```

### IP test
test your public IP(just for saving the logs files)
```bash
python util/get_pub_ip.py
```

### LLM engine test
Please run the following command to test all LLM engines:

```bash
python agentflow/scripts/test_llm_engine.py
```

Example output:
```text
🚀 Starting fault-tolerant test for 11 engines...
🧪 Testing: 'gpt-4o' | kwargs={}
✅ Success: Created ChatOpenAI
🧪 Testing: 'dashscope-qwen2.5-3b-instruct' | kwargs={}
✅ Success: Created ChatDashScope
🧪 Testing: 'gemini-1.5-pro' | kwargs={}
✅ Success: Created ChatGemini
============================================================
📋 TEST SUMMARY
============================================================
✅ Passed: 3
   • gpt-4o → ChatOpenAI
   • dashscope-qwen2.5-3b-instruct → ChatDashScope
   • gemini-1.5-pro → ChatGemini
❌ Failed: 8
   • azure-gpt-4 → 🚫 API key not found in environment
   • claude-3-5-sonnet → 🚫 API key not found in environment
   • deepseek-chat → 🚫 API key not found in environment
   • grok → 🚫 API key not found in environment
   • vllm-meta-llama/Llama-3-8b-instruct → 🚫 Connection failed
   • together-meta-llama/Llama-3-70b-chat-hf → 🚫 API key not found
   • ollama-llama3 → 🚫 Connection failed
   • unknown-model-123 → 💥 Unexpected error
============================================================
🎉 Testing complete. Script did NOT crash despite errors.
```

## Quick Start
### Train

We recommend using `tmux` to manage the training process with two windows:

1. **Create a tmux session with two windows:**
```bash
tmux new-session -s agentflow
# This creates window 0 automatically
```

2. **In Window 0 - Start the VLLM serving:**
```bash
bash train/serve_with_logs.sh
```
This will start the VLLM server to serve your base model for rollout generation during training.

3. **Create and switch to Window 1 - Start the training:**
```bash
# Press Ctrl+B then C to create a new window
# Or manually: Ctrl+B then :new-window
bash train/train_with_logs.sh
```
This will start the RL training process.

**Switching between windows:**
- `Ctrl+B` then `0` - Switch to window 0 (serving)
- `Ctrl+B` then `1` - Switch to window 1 (training)
- `Ctrl+B` then `D` - Detach from session
- `tmux attach -t agentflow` - Reattach to session

**Training hyperparameters:**

You can customize training hyperparameters according to your needs. All configurations are defined in `train/config.yaml`, including:
- Model settings (`BASE_MODEL`, `ROLLOUT_TP_SIZE`)
- Tool configurations (`ENABLE_TOOLS`, `TOOL_ENGINE`)
- Training parameters (batch size, learning rate, PPO settings)
- Data paths and lengths
- Resource allocation (GPUs, workers)

For detailed parameter descriptions and tuning options, please refer to `train/config.yaml`.

### Infer

To run inference on benchmark tasks, first ensure your model is being served via VLLM, then execute:
```bash
cd test
bash exp/run_all_models_all_datasets.sh
```

**Serving models with VLLM:**

An easy VLLM serving script can be found in `scripts/serve_vllm.sh`. This script automatically launches multiple models in parallel using tmux:

```bash
bash scripts/serve_vllm.sh
```

Before running, configure the script:
- **models**: List of model paths to serve
- **gpu_groups**: GPU allocation for each model (e.g., `"0,1"` for 2 GPUs)
- **start_port**: Starting port number (default: 8000)

The script will create a tmux session and serve each model on consecutive ports (8000, 8001, etc.) with automatic tensor parallelism based on GPU count. 

**Configuration:**
Before running, configure the script in `test/exp/run_all_models_all_datasets.sh`:
- **TASKS**: Enable/disable tasks by commenting/uncommenting (e.g., `"aime24"`, `"gameof24"`, `"bamboogle"`)
- **MODELS**: Define models with their tool configurations:
  ```bash
  MODELS=(
      "8000:vllm-IPF/AgentFlow-3B,AgentFlow-3B,Base_Generator_Tool|Python_Coder_Tool,dashscope|dashscope"
      "8001:vllm-IPF/AgentFlow-7B,AgentFlow-7B,Base_Generator_Tool|Python_Coder_Tool,dashscope|dashscope"
  )
  ```
  Format: `"port:model_path,label,tools(|-separated),engines(|-separated)"`
- **THREADS**: Number of parallel workers (default: 20)

**Results location:**
After completion, results will be organized as follows:
```
test/
└── {TASK_NAME}/           # e.g., aime24, gameof24, bamboogle
    ├── logs/
    │   └── {MODEL_LABEL}/  # e.g., AgentFlow-7B
    │       ├── 0.log       # Individual problem logs
    │       ├── 1.log
    │       └── ...
    ├── results/
    │   └── {MODEL_LABEL}/
    │       ├── finalresults_direct_output.json   # Detailed results with analysis
    │       ├── final_scores_direct_output.json   # Final scores and statistics
    │       ├── finalscore_direct_output.log      # Scoring process log
    │       ├── output_0.json              # Individual problem outputs
    │       ├── output_1.json
    │       └── ...
    └── cache/              # Cached intermediate results
```

**Key result files:**
- `final_scores_direct_output.json`: Contains accuracy, correct count, wrong PIDs, and tool usage statistics
- `finalresults_direct_output.json`: Detailed results with per-problem analysis and verification
- Individual `output_{i}.json`: Full output including query, response, memory, and execution traces

## Training Logs and Outputs

### Training Logs
During training, logs are automatically saved with IP-based organization:
```
task_logs/
└── {PUBLIC_IP}/
    └── train_log/
        ├── training_output_0000  # First 1MB of logs
        ├── training_output_0001  # Next 1MB
        ├── training_output_0002
        └── ...
```
- Logs are split into 1MB files for easier management (configurable in `train/train_with_logs.sh`)
- Maximum 5000 log files retained
- Monitor latest logs: `tail -f task_logs/{YOUR_IP}/train_log/training_output_*`

### Model Checkpoints
Trained model checkpoints are saved periodically:
```
checkpoints/
└── {PROJECT_NAME}/           # e.g., AgentFlow_general (from config.yaml)
    └── {EXPERIMENT_NAME}/    # e.g., rollout_all_7B_useklloss (from config.yaml)
        ├── global_step_2/
        │   ├── actor/
        │   │   └── huggingface/  # HuggingFace format (ready for inference)
        │   └── data.pt           # Training state
        ├── global_step_4/
        ├── global_step_6/
        └── latest_checkpointed_iteration.txt  # Points to latest checkpoint
```
**Checkpoint settings** (in `train/config.yaml`):
- `trainer.save_freq`: Checkpoint frequency (default: every 2 epochs)
- `trainer.test_freq`: Validation frequency (default: every 2 epochs)
- `trainer.total_epochs`: Total training epochs (default: 5)

### Rollout Data
During training, rollout trajectories are saved for analysis(start from 0 for each restart, the actual step may be different):
```
rollout_data/
└── {PUBLIC_IP}/
    └── {EXPERIMENT_NAME}_{TIMESTAMP}/     # e.g., rollout_all_7B_{time_stamp}
        ├── .init.lock
        ├── .run_info
        └── {MODEL_NAME}_{TIMESTAMP}/      # e.g., Qwen2.5-7B-Instruct_{time_stamp}
            ├── train/                      # Training rollouts (usually empty to save space)
            └── validation/
                ├── .val.lock
                └── step_0/                 # Validation at global step 0
                    ├── idx_0/              # Individual validation samples
                    │   └── rollout_{uuid}.json
                    ├── idx_1/
                    └── ...
```

**Rollout JSON structure** (each `rollout_{uuid}.json`):
- `prompt`: Original problem/query
- `groundtruth`: Expected answer
- `answer_extracted`: Model's extracted answer
- `reward`: Reward score (0.0 for incorrect, positive for correct)
- `total_result`: Complete execution trace including:
  - `query_analysis`: Problem analysis
  - `memory`: Step-by-step tool execution history
  - `direct_output`: Final model response
  - Tool prompts and responses for each step
- `timestamp`: Rollout generation time

**Using saved checkpoints:**
The models in `checkpoints/{PROJECT}/{EXPERIMENT}/global_step_X/actor/huggingface/` can be used for:
1. **Inference via VLLM**: Configure model paths in `scripts/serve_vllm.sh`
2. **Direct loading**: Standard HuggingFace Transformers `from_pretrained()`


