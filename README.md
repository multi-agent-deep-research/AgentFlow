# AgentFlow 

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/imgs/logo.png">
    <img alt="VerlTool" src="assets/imgs/logo.png" width=20%>
  </picture>
</p>

<h3 align="center">
AgentFlow: In-The-Flow Agentic System Optimization for Effective Planning and Tool Use.
</h3>

<p align="center">
| 
<a href=""><b>Website</b></a> |
<a href=""><b>Paper</b></a> |
<a href=""><b>Huggingface</b></a> |
<a href=""><b>Twitter</b></a> |
  <a href="https://deepwiki.com/lupantech/AgentFlow"><b>DeepWiki</b></a> |
  <a href=""><b>WeChat Group</b></a> |
  <a href=""><b>Slack</b></a>
|
</p>

## Setup
### Installation
```bash
bash setup.sh
source .venv/bin/activate
# (Optional) Install `parallel` for running benchmark experiments in parallel:
sudo apt-get update
sudo apt-get install parallel
```

### Setup Environment Variables
Duplicate the `.env.template` file and rename it to `.env`. Next, update the variables (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CX`, `DASHSCOPE_API_KEY`) with your own keys.  
```
cp .env_template .env
```

## Quick Start
### Dataset Preparation
```bash
# train data
python data/get_train_data.py
# validation data
python data/aime24_data.py
```

After that, data dir should be:
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
Start agentflow training with tmux:
```bash
# Create tmux session and start agentflow service (Window 0)
tmux new-session -s agentflow
bash train/serve_with_logs.sh

# Create new window (Ctrl+B then C) and start training (Window 1)
bash train/train_with_logs.sh
```
**Configuration:**
All training hyperparameters are in `train/config.yaml` (model settings, tools, RL parameters, resources, etc.)

### Infer
To run inference on benchmark tasks, first ensure your planner model is being served via VLLM, then execute:
```bash
cd test
bash exp/run_all_models_all_datasets.sh
```
---
























