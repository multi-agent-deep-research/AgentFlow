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