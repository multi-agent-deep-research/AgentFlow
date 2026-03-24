# BrowseComp-Plus Integration for AgentFlow

## Overview

[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) is a benchmark for evaluating **Deep Research** systems — agents that perform multi-step web search and reasoning to answer complex questions. Unlike live web search, BrowseComp-Plus uses a **fixed corpus of ~100K curated documents**, enabling fair and reproducible comparisons.

### What makes BrowseComp-Plus valuable?

- **Fixed corpus** — No variability from live web changes
- **Isolated evaluation** — Separates retriever quality from LLM reasoning
- **830 challenging queries** — Multi-hop, reasoning-intensive questions
- **Ground truth answers** — Human-verified for accurate evaluation

## How AgentFlow Uses BrowseComp-Plus

AgentFlow is a trainable agentic system with specialized modules (Planner, Executor, Verifier, Generator). The BrowseComp-Plus integration serves two purposes:

### 1. As an Evaluation Benchmark

Test how well AgentFlow performs on deep research tasks compared to other systems (OpenAI, Anthropic, Gemini, etc.).

```
Query: "What learning institution held a 3-day event in 2002?"

┌─────────────────────────────────────────────────────────────┐
│ AgentFlow                                                  │
│ ├── Planner: Break down question, search strategy        │
│ ├── Executor: Use BrowseComp search tool                │
│ └── Verifier: Cross-check information, final answer       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────────────┐
            │ BrowseComp-Plus Corpus (100K docs)  │
            │ • BM25 keyword search              │
            │ • FAISS semantic search              │
            └─────────────────────────────────────┘
```

### 2. As a Training Resource

Use BrowseComp-Plus queries as training data for Flow-GRPO:
- **Reward signal**: Accuracy of final answer
- **Tool usage**: Retrieval quality metrics
- **Long-horizon reasoning**: Multi-step planning

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      AgentFlow System                         │
│                                                               │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │ Planner │───▶│ Executor│───▶│ Verifier│───▶│ Generator│ │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘ │
│       │              │              │              │        │
│       └──────────────┴──────────────┴──────────────┘        │
│                          │                               │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
                ┌────────────────────────────────┐
                │   BrowseComp-Plus Search Tool   │
                │  (BM25 or FAISS over 100K docs)  │
                └────────────────────────────────┘
```

## Search Options

### BM25 Search (Keyword-based)

**Pros:**
- Runs on CPU (no GPU required)
- Fast, lightweight
- Good for exact keyword matches

**Cons:**
- Misses semantic matches
- Requires Java JDK 21

**Installation:**
```bash
# Install Java JDK 21
sudo apt update && sudo apt install -y openjdk-21-jdk

# Install Python dependencies
uv pip install -r requirements-browsecomp.txt

# Or install manually
uv pip install pyserini>=0.39.0 faiss-cpu>=1.13.0 peft>=0.18.0 accelerate>=0.12.0
uv pip install git+https://github.com/texttron/tevatron.git
```

### FAISS Search (Embedding-based)

**Pros:**
- Semantic understanding
- Better for paraphrases/concept queries
- No Java required

**Cons:**
- Requires GPU for embedding model
- More computationally expensive

**Installation:**
```bash
# All dependencies from requirements-browsecomp.txt
uv pip install -r requirements-browsecomp.txt

# For GPU support, replace faiss-cpu with:
uv pip install faiss-gpu>=1.13.0
```

## Quick Start

### 1. Clone AgentFlow and Set Up Base Environment

```bash
git clone https://github.com/multi-agent-deep-research/AgentFlow.git
cd AgentFlow

# Ensure uv is in PATH (pip installs it to ~/.local/bin)
export PATH=$HOME/.local/bin:$PATH

bash setup.sh
source .venv/bin/activate
git checkout feature/browsecomp-integration
```

### 2. Install BrowseComp Dependencies

```bash
# Core BrowseComp dependencies
uv pip install -r requirements-browsecomp.txt

# Additional required packages (not in base setup)
uv pip install wandb python-dotenv
uv pip install git+https://github.com/texttron/tevatron.git
uv pip install qwen-omni-utils

# Install Java JDK 21 for BM25 search
sudo apt update && sudo apt install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64  # adjust for your system
```

#### CUDA / flash-attn Setup (for FAISS search)

torch, torchvision, and flash-attn must all be compiled for the **same CUDA version**.
The base setup uses CUDA 12.8. Check your versions with `python check_env.py`.

```bash
# 1. Ensure CUDA 12.8 toolkit is available
#    If not installed system-wide, install to home dir (no sudo needed, ~4.5 GB):
#    wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
#    sh cuda_12.8.0_570.86.10_linux.run --toolkit --silent --toolkitpath=$HOME/cuda-12.8 --override
export CUDA_HOME=$HOME/cuda-12.8  # or /usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# 2. Install torch + torchvision for cu128 (if not already from setup.sh)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Build flash-attn from source (~10 min, must match torch CUDA version)
uv pip install pip  # needed for source build
python -m pip install flash-attn --no-build-isolation --no-binary flash-attn --no-cache-dir

# 4. Verify everything matches
python check_env.py
```

> **Warning:** Do NOT run `pip install searcher`. The `searcher` module comes from the
> BrowseComp-Plus repository and is loaded automatically via `sys.path`. The PyPI
> `searcher` package is a completely different project and will cause import errors.

```bash
# HuggingFace CLI (needed to download indexes)
pip install -U "huggingface_hub[cli]"
```

### 3. Clone BrowseComp-Plus and Download Indexes

```bash
# Clone BrowseComp-Plus repository (sibling to AgentFlow)
cd ..
git clone https://github.com/texttron/BrowseComp-Plus.git

# Ensure huggingface-cli is available (download script uses it)
ln -s ~/.local/bin/hf ~/.local/bin/huggingface-cli

# Download pre-built indexes
cd BrowseComp-Plus
bash scripts_build_index/download_indexes.sh
cd ../AgentFlow
```

### 4. Set Environment Variables

```bash
export BROWSECOMP_INDEX_PATH=~/BrowseComp-Plus/indexes/bm25
export BROWSECOMP_INDEX_TYPE=bm25
# export DEEPINFRA_API_KEY=your_key   # for eval
```

> **Important:** `BROWSECOMP_INDEX_PATH` must be set before running eval. Without it,
> the BrowseComp search tool will fail to load and the agent will run without search,
> resulting in near-zero accuracy. You'll see this error in the log:
> `Error instantiating BrowseComp_Search_Tool: index_path must be provided`

### 5. Run the Demo Script

The easiest way to see BrowseComp-Plus in action with AgentFlow:

```bash
# Set environment variables for the search tool
export BROWSECOMP_INDEX_PATH=/path/to/BrowseComp-Plus/indexes/bm25
export BROWSECOMP_INDEX_TYPE=bm25

# Run the quick start demo
python quick_start_browsecomp.py
```

**Expected output:**
```
======================================================================
AgentFlow Quick Start with BrowseComp-Plus
======================================================================

Query: What is the capital of France?
======================================================================

[Tool]: BrowseComp_Search_Tool
[Command]: execution = tool.execute(query="capital of France", k=5)

==> 🐙 Final Answer:
The capital of France is Paris.
```

### 6. Run Tests

```bash
# Test everything (dataset + BM25 + FAISS)
python test_browsecomp.py

# Test only dataset loading (no search)
python test_browsecomp.py --no-search

# Test only BM25
python test_browsecomp.py --index-type bm25

# Test only FAISS
python test_browsecomp.py --index-type faiss
```

### 7. Use BrowseComp Search as a Standalone Tool

```python
from agentflow.tools.browsecomp_search import BrowseComp_Search_Tool

# Initialize BM25 searcher
tool = BrowseComp_Search_Tool(
    index_type="bm25",
    index_path="BrowseComp-Plus/indexes/bm25",
    k=5
)

# Search for documents
results = tool.execute("What is the capital of France?")
print(results)
```

## Dataset Structure

| Component | Source | Size |
|-----------|--------|------|
| **Corpus** | `Tevatron/browsecomp-plus-corpus` | 100,195 documents |
| **Queries** | `Tevatron/browsecomp-plus` (encrypted) | 830 queries |
| **BM25 Index** | Pre-built Lucene index | ~50 MB |
| **FAISS Index** | Pre-built Qwen3-Embedding indexes | ~4 GB (varies by model size) |

### Document Structure

Each document contains:
```json
{
  "docid": "5412",
  "text": "---\ntitle: Arwa University holds annual cultural activities...\ndate: 2002...\n---",
  "url": "https://yementimes.com/arwa-university-holds-annual-cultural-activities..."
}
```

### Query Structure

Each query contains:
```json
{
  "query_id": "769",
  "query": "Please tell me the name of the learning institution...",
  "answer": "Queen Arwa University"
}
```

## Running Evaluation

### Prerequisites

- **2 GPUs recommended**: one for vLLM planner, one for the FAISS embedding model
- **API key**: `DEEPINFRA_API_KEY` or `OPENROUTER_API_KEY` for the executor/verifier model
- **Environment variables**:
  ```bash
  export BROWSECOMP_INDEX_PATH=/path/to/BrowseComp-Plus/indexes/faiss/qwen3-embed-8b
  export BROWSECOMP_INDEX_TYPE=faiss
  export DEEPINFRA_API_KEY=your_key
  ```

### Mode 1: Hybrid (Local Planner + API Model)

This is the recommended mode. The fine-tuned planner runs locally via vLLM, while the executor, verifier, generator, and search summarizer use an API model.

**Step 1: Start the vLLM planner server** (pick a free GPU):
```bash
cd ~/AgentFlow
source .venv/bin/activate
export CUDA_HOME=$HOME/cuda-12.8  # if installed locally
export PATH=$CUDA_HOME/bin:$PATH

# If vLLM crashes with "undefined symbol" errors, reinstall to match your torch version:
# uv pip install vllm --force-reinstall

# Start vLLM on a specific GPU (adjust GPU index and memory utilization as needed)
# Foreground:
CUDA_VISIBLE_DEVICES=3 vllm serve AgentFlow/agentflow-planner-7b --port 8000 --gpu-memory-utilization 0.3
# Or background:
nohup bash -c 'CUDA_VISIBLE_DEVICES=3 vllm serve AgentFlow/agentflow-planner-7b --port 8000 --gpu-memory-utilization 0.3' > vllm.log 2>&1 &
```

Wait until you see `Application startup complete`, or check with:
```bash
curl -s http://localhost:8000/v1/models
```

**Step 2: Run the evaluation** (in a separate terminal, pick a different GPU for FAISS):
```bash
cd ~/AgentFlow
source .venv/bin/activate
export CUDA_HOME=$HOME/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

CUDA_VISIBLE_DEVICES=2 python run_browsecomp_eval.py \
    --service deepinfra \
    --model Qwen/Qwen3-14B \
    --num-queries 50 \
    --max-steps 20
```

### Mode 2: Unified (Single API Model for Everything)

No local GPU needed for the planner — all components use the same API model.

```bash
python run_browsecomp_eval.py \
    --unified \
    --service deepinfra \
    --model zai-org/GLM-4.7-Flash \
    --num-queries 50 \
    --max-steps 5
```

### Mode 3: Parallel (Shared Embedding Server)

For faster evaluation, run multiple workers sharing a single embedding server.
Workers don't need GPU — only the embedding server and vLLM LLM servers do.

**Step 1: Start the embedding server** (one GPU, once):
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-8B \
    --port 8002 --task embed \
    --override-pooler-config '{"pooling_type": "LAST", "normalize": true}' \
    --gpu-memory-utilization 0.3
```

**Step 2: Run parallel eval:**
```bash
export EMBEDDING_API_BASE=http://localhost:8002/v1
export BROWSECOMP_INDEX_PATH=~/BrowseComp-Plus/indexes/qwen3-embedding-8b
export BROWSECOMP_INDEX_TYPE=faiss
export HOSTED_VLLM_API_BASE=http://localhost:8001/v1

bash run_browsecomp_parallel.sh --workers 4 \
    --service hosted_vllm --model Qwen/Qwen3.5-9B \
    --max-steps 20 --output-dir runs/agentflow_parallel
```

Workers automatically skip completed queries, so they load-balance across the dataset.
No GPU needed per worker — the FAISS index stays on CPU while embeddings are computed
by the shared vLLM server.

### Example Setups

#### Setup A: Fine-tuned planner + Qwen3.5-9B (hybrid)

The fine-tuned planner produces cleaner tool calls and is faster. Recommended for best results.

```
GPU 0: Planner (agentflow-planner-7b, 14GB) + Embeddings (Qwen3-Embedding-8B, 16GB) = ~30GB
GPU 1: Executor (Qwen3.5-9B, 18GB)
```

```bash
# Start servers
CUDA_VISIBLE_DEVICES=0 vllm serve AgentFlow/agentflow-planner-7b --port 8000 --gpu-memory-utilization 0.3 &
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-8B --port 8002 --task embed \
    --override-pooler-config '{"pooling_type": "LAST", "normalize": true}' --gpu-memory-utilization 0.35 &
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.5-9B --port 8001 --gpu-memory-utilization 0.4 &

# Wait for all servers
for port in 8000 8001 8002; do
    while ! curl -s http://localhost:$port/v1/models > /dev/null 2>&1; do sleep 5; done
    echo "Port $port ready"
done

# Run eval
export EMBEDDING_API_BASE=http://localhost:8002/v1
export HOSTED_VLLM_API_BASE=http://localhost:8001/v1
export BROWSECOMP_INDEX_PATH=~/BrowseComp-Plus/indexes/qwen3-embedding-8b
export BROWSECOMP_INDEX_TYPE=faiss

python run_browsecomp_eval.py --service hosted_vllm --model Qwen/Qwen3.5-9B \
    --max-steps 20 --output-dir runs/setup_a_planner7b_qwen35

# Or parallel (4 workers, no extra GPU needed):
bash run_browsecomp_parallel.sh --workers 4 \
    --service hosted_vllm --model Qwen/Qwen3.5-9B \
    --max-steps 20 --output-dir runs/setup_a_planner7b_qwen35
```

#### Setup B: Qwen3.5-9B for everything (unified)

Same model for planner + executor. Simpler (2 servers instead of 3) but the planner
is less reliable at tool selection — may hallucinate tool names.

```
GPU 0: Executor+Planner (Qwen3.5-9B, 18GB) + Embeddings (Qwen3-Embedding-8B, 16GB) = ~34GB
GPU 1: free (or use for another experiment)
```

```bash
# Start servers
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-9B --port 8001 --gpu-memory-utilization 0.4 &
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-8B --port 8002 --task embed \
    --override-pooler-config '{"pooling_type": "LAST", "normalize": true}' --gpu-memory-utilization 0.35 &

# Wait for servers
for port in 8001 8002; do
    while ! curl -s http://localhost:$port/v1/models > /dev/null 2>&1; do sleep 5; done
    echo "Port $port ready"
done

# Run eval (--unified uses the same model for planner)
export EMBEDDING_API_BASE=http://localhost:8002/v1
export HOSTED_VLLM_API_BASE=http://localhost:8001/v1
export BROWSECOMP_INDEX_PATH=~/BrowseComp-Plus/indexes/qwen3-embedding-8b
export BROWSECOMP_INDEX_TYPE=faiss

python run_browsecomp_eval.py --unified --service hosted_vllm --model Qwen/Qwen3.5-9B \
    --max-steps 20 --output-dir runs/setup_b_qwen35_unified
```

### GPU Assignment Reference

| Component | Model | VRAM | Port |
|-----------|-------|------|------|
| Planner (fine-tuned) | AgentFlow/agentflow-planner-7b | ~14 GB | 8000 |
| Executor / Planner (unified) | Qwen/Qwen3.5-9B | ~18 GB | 8001 |
| Embeddings (FAISS search) | Qwen/Qwen3-Embedding-8B | ~16 GB | 8002 |
| Eval workers | — | No GPU | — |

### Key Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--service` | API provider (`deepinfra` or `openrouter`) | `openrouter` |
| `--model` | Model name on the service | `qwen/qwen-2.5-7b-instruct` |
| `--unified` | Use API model for all components (no vLLM) | `False` |
| `--num-queries` | Number of queries to evaluate | all (830) |
| `--max-steps` | Maximum reasoning steps per query | 3 |
| `--random` | Randomly sample queries instead of first N | `False` |
| `--judge-model` | Model for LLM-as-judge evaluation | `openai/gpt-4.1` |

> **Note on judge model:** When using `--service hosted_vllm`, the judge defaults to the
> same local model (e.g. Qwen3.5-9B), which produces high parse error rates (~40%) and
> unreliable accuracy numbers. For proper evaluation, use a stronger judge model via an API:
> ```bash
> # Use DeepSeek-V3 as judge while running agents on local vLLM
> export DEEPINFRA_API_KEY=your_key
> python run_browsecomp_eval.py --service hosted_vllm --model Qwen/Qwen3.5-9B \
>     --judge-model deepseek-ai/DeepSeek-V3
> ```

### Search Result Summarization

The BrowseComp search tool uses LLM summarization to compress retrieved documents into focused summaries (instead of truncating at 512 characters). This preserves key facts like names, dates, numbers, and DocIDs. The summarization model is the same as `--model`. If summarization fails (e.g., context too long), it falls back to truncated snippets.

### Output

Results are saved to `runs/agentflow/<timestamp>/`:
- `eval.log` — full execution log
- `<query_id>.json` — per-query results
- `judge_results.json` — LLM judge scores
- `summary.json` — aggregate metrics

Runs are also logged to W&B (`AgentFlow-Pro-Eval-BrowseComp-Plus` project).

## Analyzing Results

Use `analyze_wandb_run.py` to compute statistics from evaluation runs. It works with W&B runs (downloads artifacts automatically) and local result directories.

To download from W&B, add your API key to `.env`:
```
WANDB_API_KEY=your_wandb_api_key
```

### Usage

```bash
# From a W&B URL
python analyze_wandb_run.py https://wandb-radfan.ru/multiagent-deepresearch-improvement/AgentFlow-Pro-Eval-BrowseComp-Plus/runs/h4dh79yg/overview

# From a W&B run ID
python analyze_wandb_run.py h4dh79yg

# From a local results directory
python analyze_wandb_run.py runs/agentflow_planner7b_qwen35/20260318_144941

# Compare multiple runs side by side
python analyze_wandb_run.py h4dh79yg abc123de runs/local_dir

# Output as JSON
python analyze_wandb_run.py h4dh79yg --json
```

### What it computes

| Category | Metrics |
|----------|---------|
| **Queries** | Total, completed, failed, completion rate |
| **Tool calls** | Avg/median/max per query, search vs generate split, distribution |
| **Context length** | Avg/median/max in chars and estimated tokens (~4 chars/token) |
| **Steps** | Avg/median/max reasoning steps per query |
| **Execution time** | Avg/median per query, total runtime |
| **Retrieved docs** | Avg per query, total unique DocIDs |

### How it works

1. **W&B runs:** Downloads the `evaluation_results` artifact to `wandb_cache/<run_id>/` (cached — won't re-download on subsequent runs)
2. **Local dirs:** Reads `<query_id>.json` files directly from the results directory (handles nested timestamp subdirectories automatically)
3. **Context estimation:** For each query, accumulates the query text + all step sub-goals, commands, and results to estimate the total context window used. Tokens are estimated at ~4 chars/token

### Example output

```
--- Tool Calls ---
  Avg per query:  7.7
  Avg search:     7.2
  Avg generate:   0.4

--- Context Length ---
  Avg (chars):    20,136
  Max (chars):    58,494
  Avg (~tokens):  5,034
  Max (~tokens):  14,623

--- Steps ---
  Avg:            7.7
  Max:            10
```

When comparing multiple runs, a side-by-side table is printed at the end.

## Evaluation Format

To evaluate AgentFlow on BrowseComp-Plus, format results as:

```json
{
  "query_id": "769",
  "tool_call_counts": {
    "BrowseComp_Search_Tool": 3
  },
  "status": "completed",
  "retrieved_docids": ["5412", "12345", "67890"],
  "result": [{
    "type": "output_text",
    "output": "Explanation: Based on the documents [5412], the institution is Queen Arwa University.\nExact Answer: Queen Arwa University\nConfidence: 95%"
  }]
}
```

Then use BrowseComp-Plus's evaluation script (LLM-as-a-judge) to score accuracy.

## Why This Matters for AgentFlow

1. **Reproducible Benchmarking** — Compare AgentFlow against other systems fairly
2. **Tool Quality Measurement** — Is AgentFlow retrieving the right documents?
3. **Long-Form Reasoning** — Can AgentFlow synthesize information across multiple sources?
4. **Training Signal** — Use accuracy as reward for Flow-GRPO optimization

## Links

- [BrowseComp-Plus Paper](https://arxiv.org/pdf/2508.06600)
- [BrowseComp-Plus GitHub](https://github.com/texttron/BrowseComp-Plus)
- [Leaderboard](https://huggingface.co/spaces/Tevatron/BrowseComp-Plus)
- [AgentFlow Paper](https://arxiv.org/abs/2510.05592)
