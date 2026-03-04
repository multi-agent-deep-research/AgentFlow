# Instructions from Ignat
## Setup
- follow the original instructions for setup, you only need:
  - YANDEX_SEARCH_FOLDER_ID
  - YANDEX_SEARCH_API_KEY
  - YANDEX_SEARCH_TYPE
  - YANDEX_SEARCH_POLL_INTERVAL
  - TOGETHER_API_KEY (or OPENAI_API_KEY; see [1] bellow)

  the former are used for web search and the latter for embeddings 
- adjust [`train/config.yaml`](train/config.yaml) and [`scripts/serve_vllm_qwen.sh`](scripts/serve_vllm_qwen.sh) for your gpu configuration
- [1] if want to use openai model for embeddings change TOGETHER_EMBEDDING_MODEL in [`wetup_env.sh`](wetup_env.sh) to `text-embedding-3-small` or `text-embedding-3-large`

## Train
```shell
source .venv/bin/activate
source setup_env.sh
```
Create vllm instance for tools and other agents (if you use vllm)
```shell
bash scripts/serve_vllm_qwen.sh
```
Start agentflow training using Flow-GRPO with tmux:
```bash
# Create tmux session and start agentflow service (Window 0)
tmux new-session -s agentflow
bash train/serve_with_logs.sh

# Create new window (Ctrl+B then C) and start training (Window 1)
bash train/train_with_logs.sh
```
# End of Instructions from Ignat

<a name="readme-top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/img/logo.png">
    <img alt="AgentFlow" src="assets/img/logo.png" width=31%>
  </picture>
</p>

<h3 align="center">
AgentFlow: In-the-Flow Agentic System Optimization
</h3>


<!--- BADGES: START --->
<p align="center">
    <a href="https://arxiv.org/abs/2510.05592"><img src="https://img.shields.io/badge/arXiv-2510.05592-B31B1B.svg?logo=arxiv" alt="Arxiv"></a>
    <a href="https://huggingface.co/spaces/AgentFlow/agentflow"><img src="https://img.shields.io/badge/Gradio-Demo-F97316.svg?logo=gradio" alt="Gradio Demo"></a>
    <a href="https://huggingface.co/papers/2510.05592"><img src="https://img.shields.io/badge/Huggingface-Paper-FFD21E.svg?logo=huggingface" alt="Huggingface Paper"></a>
    <a href="https://huggingface.co/AgentFlow"><img src="https://img.shields.io/badge/Huggingface-Model-FFD21E.svg?logo=huggingface" alt="Huggingface Model"></a>
    <a href="https://agentflow.stanford.edu/"><img src="https://img.shields.io/badge/Website-AgentFlow-E5426E?logo=kashflow" alt="Website"></a>
    <a href="https://x.com/lupantech/status/1976016000345919803"><img src="https://img.shields.io/badge/Coverage-AgentFlow-2176BC.svg?logo=x" alt="X"></a>
    <a href="https://www.youtube.com/watch?v=kIQbCQIH1SI"><img src="https://img.shields.io/badge/YouTube-Tutorial-FF0000?logo=youtube" alt="Youtube"></a>
    <a href="https://deepwiki.com/lupantech/AgentFlow"><img src="https://img.shields.io/badge/DeepWiki-AgentFlow-6B4FBB?logo=readthedocs&logoColor=white" alt="DeepWiki"></a>
    <a href="https://join.slack.com/t/agentflow-co/shared_invite/zt-3f712xngl-LfxS4gmftAeKvcxR3nSkWQ"><img src="https://img.shields.io/badge/Slack-AgentFlow-D41544.svg?logo=slack" alt="Slack"></a>
    <a href="https://github.com/lupantech/AgentFlow/blob/main/assets/img/wechat_group.jpg">
  <img src="https://img.shields.io/badge/Wechat-AgentFlow-07C160.svg?logo=wechat" alt="Wechat AgentFlow">
</a>
  
  </p>
<!--- BADGES: END --->


## 📣 News
- **[2025.10.26]** 📚 Our project introduction has been featured on **[DeepWiki](https://deepwiki.com/lupantech/AgentFlow)**!
- **[2025.10.16]** 🏆 Our paper has been accepted by [**NeurIPS 2025 Efficient Reasoning Workshop**](https://efficient-reasoning.github.io/)!
- **[2025.10.13]** 📸 Excited to have a tutorial video for AgentFlow covered by Discover AI on **[YouTube](https://www.youtube.com/watch?v=kIQbCQIH1SI)**!
- **[2025.10.10]** 🚀 Our X [post](https://x.com/lupantech/status/1976016000345919803) received **1K+ likes**! Feel free to check out the post and join the discussion! 💬
- **[2025.10.08]** 🔥 We are honored to be featured as 🤗 HuggingFace **[Daily Paper #2](https://huggingface.co/papers/2510.05592)**.

## 🌟 Why AgentFlow?
AgentFlow is a **trainable, tool-integrated agentic framework** designed to overcome the **scalability** and **generalization limits** of today’s tool-augmented reasoning approaches. 

Unlike prevailing approaches such as [Search-R1](https://github.com/PeterGriffinJin/Search-R1) which train a **single LLM** to interleave reasoning steps with tool calls, **AgentFlow** introduces a **modular agentic system** with four specialized modules: 🧭 **Planner**, 🛠 **Executor**, ✅ **Verifier**, and ✍️ **Generator**.

![framework_overall](assets/img/framework.png)

For effective planning and tool use, the framework directly **optimizes planner agent within the system** in an **online fashion** using **Flow-based Group Refined Policy Optimization (Flow-GRPO)**, achieving superior performance across diverse domains with improved tool-calling reliability and long-horizon reasoning capabilities.

![flow_grpo](assets/img/flow_grpo.png)

## 📺 YouTube Tutorial
Excited to have a tutorial video for AgentFlow covered by [Discover AI](https://www.youtube.com/@code4AI) on YouTube!

<!-- [![AgentFlow Tutorial](https://img.youtube.com/vi/kIQbCQIH1SI/0.jpg)](https://www.youtube.com/watch?v=kIQbCQIH1SI) -->

<div align="center">
  <a href="https://www.youtube.com/watch?v=kIQbCQIH1SI">
    <img src="https://img.youtube.com/vi/kIQbCQIH1SI/maxresdefault.jpg" alt="AgentFlow Tutorial" width="100%">
  </a>
</div>


## 🚀 Key Features

- 🧩 **Modular Agentic System** – Four specialized agent modules (**Planner**, **Executor**, **Verifier**, **Generator**) that coordinate via evolving memory and integrated tools across multiple turns.  
- 🔗 **Multi-Tool Integration** – Seamlessly connect with diverse tool ecosystems, including `base_generator`, `python_coder`, `google_search`, `wikipedia_search`, `web_search`, and more.  
- 🎯 **Flow-GRPO Algorithm** – Enables **in-the-flow agent optimization** for **long-horizon reasoning tasks** with sparse rewards.
- 📈 **Proven Results** – **AgentFlow (7B Backbone)** beats top baselines on 10 benchmarks, with **+14.9% search**, **+14.0% agentic**, **+14.5% math**, **+4.1% science**, even outperforming ~200B-parameter **GPT-4o**.

---

## 📑 Table of Contents
- [⚙️ Setup](#️-setup)
  - [Installation](#installation)
  - [Setup Environment Variables](#setup-environment-variables)
- [⚡ Quick Start on AgentFlow Inference](#-quick-start-on-agentflow-inference)
- [💥 Quick Start on AgentFlow Flow-GRPO Training](#-quick-start-on-agentflow-flow-grpo-training)
  - [(Optional) Test Your Environment](#optional-test-your-environment)
  - [Dataset Preparation](#dataset-preparation)
  - [Flow-GRPO Training](#flow-grpo-training)
- [🎯 AgentFlow Benchmark](#-agentflow-benchmark)
- [🧩 Use Your Own Model in AgentFlow](#-use-your-own-model-in-agentflow)
- [🤝 Core Contributors](#-core-contributors)
- [🎓 Advisors](#-advisors)
- [🙏 Acknowledgements](#-acknowledgements)
- [🚀 Contributing](#-contributing)

## ⚙️ Setup

### Prerequisites
- **Python 3.11** (recommended)

### Installation
```bash
bash setup.sh
source .venv/bin/activate
# (Optional) Install `parallel` for running benchmark experiments in parallel:
sudo apt-get update
sudo apt-get install parallel
```

### Setup Environment Variables
Copy the `.env.template` file from `agentflow/.env.template` and rename it to `.env`, then place it in the `agentflow/` folder. Update the following variables with your own API keys:
- `OPENAI_API_KEY` (for judging reasponse)
- `GOOGLE_API_KEY` (for Google Search tool)
- `DASHSCOPE_API_KEY` ([optional] for calling Qwen-2.5-7B-Instruct as engine for agents and tools)
- `TOGETHER_API_KEY` ([optional] alternative for calling Qwen-2.5-7B-Instruct as engine for agents and tools - recommended for international users)
- More ways: serve Qwen2.5-7B-instruct model with vLLM (details refer to [`serve_vllm_local.md`](assets/doc/serve_vllm_local.md)).

Please check [API Key Setup Guide](assets/doc/api_key.md) for detailed instructions on how to obtain these keys.

```bash
cp agentflow/.env.template agentflow/.env
# Then edit agentflow/.env with your API keys
```

## 🔍 Check Before You Run (Recommended)
Before running inference or training, we recommend verifying that your API keys and environment are properly configured.

### 🛠️ Test Tools
Run the following command to test all integrated tools:
```bash
cd agentflow/agentflow
bash ./tools/test_all_tools.sh
```
Example output:
```text
Testing all tools...
✅ base_generator passed
✅ google_search passed
✅ python_coder passed
✅ wikipedia_search passed
...
✅ All tests passed
```

### 🧠 Test LLM Engines
Verify that your LLM engines (OpenAI, DashScope, Gemini, etc.) are correctly initialized and responding:
```bash
python agentflow/scripts/test_llm_engine.py
```
Example output:
```text
🚀 Starting fault-tolerant test for 11 engines...
✅ Passed: 4
   • gpt-4o → ChatOpenAI
   • dashscope-qwen2.5-3b-instruct → ChatDashScope
   • gemini-1.5-flash → ChatGemini
   • deepseek-chat → ChatDeepseek
...
🎉 All engines initialized successfully!
```

## ⚡ Quick Start on AgentFlow Inference 
AgentFlow provides a modular agentic system with **four specialized modules** (planner, executor, verifier, generator) that coordinate through **evolving memory** and a **toolkit** over **multiple turns** to solve complex reasoning tasks. 

To quickly experience the system in action, run the command below (don’t forget to set up your API key):
```bash 
python quick_start.py
```
Example output of `python quick_start.py`:
```text
==> Initializing agentflow...
==> Setting up tools...
==> 🎯 Reasoning Steps from AgentFlow (Deep Thinking...)
==> 🔍 Step 0: Query Analysis
==> 🎯 Step 1: Action Prediction (Google_Search_Tool)
==> 🛠️ Step 1: Command Execution (Google_Search_Tool)
...
**Answer:** The capital of France is Paris.
==> ✅ Query Solved!

**Process Summary:**
1. **Query Analysis:** Identified as a factual question about the capital of France.
2. **Tool Selection:** Used Google Search for accurate information.
3. **Execution:** Confirmed Paris as the capital.
4. **Verification:** Cross-referenced sources for reliability.

**Answer:** The capital of France is Paris.
```

## 💥 Quick Start on AgentFlow Flow-GRPO Training 
For effective planning and tool use, the framework directly **optimizes the planner agent within the system in an online fashion using Flow-GRPO**. Below is a quick start for training.

### (Optional) Test Your Environment
Before diving in, we recommend verifying that AgentFlow's tools, LLM engines, and network configuration are properly set up. See [test_env.md](assets/doc/test_env.md) for detailed testing instructions.


### Dataset Preparation
We mix two datasets for training: [NQ (Natural Questions)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) for agentic search and [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) for mathematical reasoning.

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

### Flow-GRPO Training 
Start agentflow training using Flow-GRPO with tmux:
```bash
# Create tmux session and start agentflow service (Window 0)
tmux new-session -s agentflow
bash train/serve_with_logs.sh

# Create new window (Ctrl+B then C) and start training (Window 1)
bash train/train_with_logs.sh
```

**Configuration:**
All training hyperparameters are in [`train/config.yaml`](train/config.yaml) (model settings, tools, RL parameters, resources, etc.)

**Logging:**
We provide a comprehensive logging to monitor training. See [logs.md](assets/doc/logs.md) for more details.



## 🎯 AgentFlow Benchmark
Serve the trained planner model with VLLM (here we deploy our [7B Flow-GRPO planner model](https://huggingface.co/AgentFlow/agentflow-planner-7b)):
```bash
bash scripts/serve_vllm.sh
```

Run inference on specific benchmark tasks:
```bash
cd test
# Run Bamboogle benchmark
bash bamboogle/run.sh
```

After running, each task folder (e.g., `test/bamboogle/`) will contain:
- `data/`: Contains the evaluation dataset (e.g., `data.json`).
- `logs/`: Contains detailed execution logs for each problem index (organized by model label).
- `results/`: Contains the model's generated answers (`output_i.json`) and final evaluation scores (`finalscore_*.log`).

You can find more benchmarking details in [benchmark.md](assets/doc/benchmark.md).

## 🔬 BrowseComp-Plus Evaluation

[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) is a benchmark for evaluating Deep Research agents with a fixed corpus of ~100K curated documents. AgentFlow now supports evaluation on this benchmark.

### Dependencies

**For BM25 search (requires Java JDK):**
```bash
# Option 1: Install Java JDK 21 via conda (recommended)
conda install -c conda-forge openjdk=21

# Option 2: Install via system package manager
sudo apt-get install -y openjdk-21-jdk

# Set JVM path (if using conda)
export JVM_PATH=$HOME/miniconda3/lib/jvm/lib/server/libjvm.so

# Install Python packages
pip install pyserini>=1.2.0
pip install git+https://github.com/texttron/tevatron.git
pip install qwen-omni-utils
```

**For FAISS search (no Java required):**
```bash
pip install faiss-cpu>=1.11.0.post1
pip install git+https://github.com/texttron/tevatron.git
pip install qwen-omni-utils
```

**Common dependencies (both BM25 and FAISS):**
```bash
pip install datasets>=4.0.0 tqdm>=4.67.1
```

**Quick install (all at once):**
```bash
pip install -r requirements-browsecomp.txt
# Then install Java (for BM25) or tevatron dependencies:
conda install -c conda-forge openjdk=21  # for BM25 only
pip install git+https://github.com/texttron/tevatron.git qwen-omni-utils
```

### Setup

1. **Download BrowseComp-Plus** (if not already present):
```bash
# Clone BrowseComp-Plus in the parent directory
cd ..
git clone https://github.com/texttron/BrowseComp-Plus.git
cd AgentFlow
```

2. **Download the index**:
```bash
cd ../BrowseComp-Plus
bash scripts_build_index/download_indexes.sh
cd ../AgentFlow
```

3. **Decrypt queries**:
```bash
cd ../BrowseComp-Plus
python scripts_build_index/decrypt_dataset.py \
    --output data/browsecomp_plus_decrypted.jsonl \
    --generate-tsv topics-qrels/queries.tsv
cd ../AgentFlow
```

### Run Evaluation (Test with few samples)

```bash
# Test with just 5 queries
python -m agentflow.evaluations browsecomp \
    --index-path ../BrowseComp-Plus/indexes/bm25_index \
    --index-type bm25 \
    --output-dir runs/agentflow_browsecomp_test \
    --num-queries 5 \
    --max-steps 3
```

### Run Full Evaluation

```bash
python -m agentflow.evaluations browsecomp \
    --index-path ../BrowseComp-Plus/indexes/bm25_index \
    --index-type bm25 \
    --output-dir runs/agentflow_browsecomp \
    --max-steps 3
```

### Format Results for Leaderboard

```bash
python -m agentflow.evaluations browsecomp \
    --format-only \
    --output-dir runs/agentflow_browsecomp \
    --retriever-name "AgentFlow-BM25" \
    --link "https://github.com/multi-agent-deep-research/AgentFlow"
```

### Using BrowseComp-Plus as a Tool

You can also use BrowseComp-Plus search as a tool during training/inference:

```python
from agentflow.tools.browsecomp_search import BrowseComp_Search_Tool

# Initialize the tool
tool = BrowseComp_Search_Tool(
    index_type="bm25",
    index_path="../BrowseComp-Plus/indexes/bm25_index",
    k=5
)

# Search
results = tool.execute(query="What is the capital of France?")
``` 

## 🧩 Use Your Own Model in AgentFlow

AgentFlow supports different LLM engines for each agent module. See [llm_engine.md](assets/doc/llm_engine.md) for supported models and [`factory.py`](agentflow/agentflow/engine/factory.py) for the corresponding `model_string` configuration:

**Planner Agent:**
- Modify the `llm_engine_name` parameter in the corresponding `run.sh` script (e.g., `test/bamboogle/run.sh`)

**Other Agents (Executor, Verifier, Generator):**
- By default, these agents use a fixed LLM engine (Qwen-2.5-7B-Instruct via DashScope)
- To use your own model, modify `self.llm_engine_fixed` in [`agentflow/agentflow/models/planner.py:19`](agentflow/agentflow/models/planner.py#L19):
```python
self.llm_engine_fixed = create_llm_engine(model_string="your-engine", is_multimodal=False, temperature=temperature)
```
and

- Modify the `llm_engine_name` parameter in the Executor instantiation from [`agentflow/agentflow/solver.py:232`](agentflow/agentflow/solver.py#L232):
```python
# Instantiate Executor
executor = Executor(
    # llm_engine_name=llm_engine_name,
    llm_engine_name="dashscope",
    root_cache_dir=root_cache_dir,
    verbose=verbose,
    # base_url=base_url,
    temperature=temperature
)
```
- For detailed information on supported engines and `model_string` formats, see [`llm_engine.md`](assets/doc/llm_engine.md)

## 🏆 Experiments

### 📊 Main Results
**AgentFlow (Qwen-2.5-7B-Instruct Backbone)** outperforms top baselines on 10 benchmarks:  
- **+14.9%** on search  
- **+14.0%** on agentic reasoning  
- **+14.5%** on math  
- **+4.1%** on science  

💡 Even surpasses larger proprietary models like **GPT-4o (~200B)**.

![main_table1](assets/img/maintable1.png)
![main_table2](assets/img/maintable2.png)

### 🔍 In-Depth Analysis
- Improved planning and decision-making  
- Enhanced tool-calling reliability  
- Positive scaling trends with model size & reasoning turns  

Explore more in our [paper](https://arxiv.org/abs/2510.05592) or [project page](https://agentflow.stanford.edu/).

![tool_call](assets/img/tool_call.png)

---

## 🤝 Core Contributors

<table>
<tr>
    <td align="center">
        <a href="https://zhuofeng-li.github.io/">
            <img src="https://github.com/Zhuofeng-Li.png" width="75px;" alt="Zhuofeng Li"/>
            <br />
            <sub><b>Zhuofeng Li</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://isaacghx.github.io/about/">
            <img src="https://github.com/IsaacGHX.png" width="75px;" alt="Haoxiang Zhang"/>
            <br />
            <sub><b>Haoxiang Zhang</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://lupantech.github.io/">
            <img src="https://github.com/lupantech.png" width="75px;" alt="Pan Lu"/>
            <br />
            <sub><b>Pan Lu</b></sub>
        </a>
    </td>
</tr>
</table>

## 🎓 Advisors

<table>
<tr>
    <td align="center">
        <a href="https://www.james-zou.com/">
            <img src="https://static.wixstatic.com/media/0f3e8f_cfa7e327b97745ddb8c4a66454b5eb3e~mv2.jpg/v1/fill/w_398,h_557,al_c,q_80,usm_0.66_1.00_0.01,enc_avif,quality_auto/46824428A5822_ForWeb.jpg" width="65px;" alt="James Zou"/>
            <br />
            <sub><b>James Zou</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://yejinc.github.io/">
            <img src="https://yejinc.github.io/profile-uw-2022.jpeg" width="75px;" alt="Yejin Choi"/>
            <br />
            <sub><b>Yejin Choi</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://yuzhimanhua.github.io/">
            <img src="https://yuzhimanhua.github.io/profile_pic.jpg" width="90px;" alt="Yu Zhang"/>
            <br />
            <sub><b>Yu Zhang</b></sub>
        </a>
    </td>
</tr>
</table>

## 🙏 Acknowledgements

We thank the following open-source projects:
- [verl](https://github.com/volcengine/verl) for the excellent RL framework design.
- [vLLM ](https://github.com/vllm-project/vllm) for fast LLM inference support.
- [Verl-Tool](https://github.com/TIGER-AI-Lab/verl-tool) and [agent-lightning](https://github.com/microsoft/agent-lightning) for their early-stage exploration in agentic RL Training. 

We thank [Lambda](https://lambda.ai/careers) for GPU support!

## 🚀 Contributing

We are truly looking forward to open-source contributions to AgentFlow!  If you’re interested in contributing, collaborating, or reporting issues, please feel free to open an issue or submit a pull request (PR).  You can also reach us at [zhuofengli12345@gmail.com](mailto:zhuofengli12345@gmail.com), [isaacpfino@gmail.com](mailto:isaacpfino@gmail.com), [lupantech@gmail.com](mailto:lupantech@gmail.com) or join our Slack community: [AgentFlow](https://join.slack.com/t/agentflow-co/shared_invite/zt-3f712xngl-LfxS4gmftAeKvcxR3nSkWQ).


We are also looking forward to your feedback and suggestions!

## 📚 Citation
```bibtex
@article{li2025flow,
  title={In-the-Flow Agentic System Optimization for Effective Planning and Tool Use},
  author={Li, Zhuofeng and Zhang, Haoxiang and Han, Seungju and Liu, Sheng and Xie, Jianwen and Zhang, Yu and Choi, Yejin and Zou, James and Lu, Pan},
  journal={arXiv preprint arXiv:2510.05592},
  year={2025}
}
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lupantech/AgentFlow&type=Date)](https://star-history.com/#lupantech/AgentFlow&Date)

<p align="right" style="font-size: 14px; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>
