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
