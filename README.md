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
<a href="https://huggingface.co/agentflow"><b>Huggingface</b></a> |
<a href=""><b>Twitter</b></a> |
  <a href="https://deepwiki.com/lupantech/AgentFlow"><b>DeepWiki</b></a> |
  <a href=""><b>WeChat Group</b></a> |
  <a href=""><b>Slack</b></a>
|
</p>

AgentFlow is a trainable, tool-integrated agentic framework that addresses the scalability and generalization limitations of current tool-augmented approaches. 


Unlike prevailing methods, like Search-R1 that train a single LLM interleaving thoughts and tool calls, **AgentFlow provides a modular agentic system with four specialized modules (planner, executor, verifier, generator) that coordinate through evolving memory and a toolkit over multiple turns** to solve complex reasoning tasks. For effective planning and tool use, the framework directly **optimizes the planner agent within the system in an online fashion using Flow-based Group Refined Policy Optimization (Flow-GRPO)**, achieving superior performance across diverse domains with improved tool-calling reliability and long-horizon reasoning capabilities.

## Key Features
+ **Modular Agentic System**: Powerful Agentic System with four specialized modules (planner, executor, verifier, generator) augmented with tools that coordinate through evolving memory across multiple turns.
+ **Multi-Tool Integration**: Seamless integration with diverse tool ecosystems, including base_generator, python_coder, google_search, wikipedia_search, and web_search.
+ **Flow-GRPO Algorithm**: Novel Flow-based Group Refined Policy Optimization that enables in-the-flow optimization within agentic systems under long-horizon reasoning tasks with sparse reward.

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
### AgentFlow Train
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

### AgentFlow Infer
Serve the trained planner model with VLLM (here we deploy our 7B Flow-GRPO planner model).:
```bash
bash scripts/serve_vllm.sh
```

Run inference on benchmark tasks:
```bash
cd test
bash exp/run_all_models_all_datasets.sh
```
























