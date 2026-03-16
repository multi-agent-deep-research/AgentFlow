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

# Install flash-attn for FAISS search (requires CUDA toolkit, ~10 min build)
uv pip install pip  # needed for source build
python -m pip install flash-attn --no-build-isolation --force-reinstall --no-cache-dir --no-binary :all:
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

**Step 1: Start the vLLM planner server** (on GPU 1):
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve AgentFlow/agentflow-planner-7b --port 8000 --gpu-memory-utilization 0.9
```

Wait until you see `Application startup complete`, or check with:
```bash
curl -s http://localhost:8000/v1/models
```

**Step 2: Run the evaluation** (FAISS searcher on GPU 0):
```bash
CUDA_VISIBLE_DEVICES=0 python run_browsecomp_eval.py \
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

### GPU Assignment

When running in hybrid mode, the vLLM planner and FAISS embedding model need separate GPUs:

| Component | GPU | How to set |
|-----------|-----|------------|
| vLLM planner | GPU 1 | `CUDA_VISIBLE_DEVICES=1 vllm serve ...` |
| FAISS searcher (embedding model) | GPU 0 | `CUDA_VISIBLE_DEVICES=0 python run_browsecomp_eval.py ...` |
| Executor/Verifier/Generator | API | Via `--service` and `--model` flags |

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

### Search Result Summarization

The BrowseComp search tool uses LLM summarization to compress retrieved documents into focused summaries (instead of truncating at 512 characters). This preserves key facts like names, dates, numbers, and DocIDs. The summarization model is the same as `--model`. If summarization fails (e.g., context too long), it falls back to truncated snippets.

### Output

Results are saved to `runs/agentflow/<timestamp>/`:
- `eval.log` — full execution log
- `<query_id>.json` — per-query results
- `judge_results.json` — LLM judge scores
- `summary.json` — aggregate metrics

Runs are also logged to W&B (`AgentFlow-Pro-Eval-BrowseComp-Plus` project).

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
