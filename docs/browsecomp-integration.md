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
conda install -c conda-forge openjdk=21
pip install pyserini tevatron qwen-omni-utils
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
pip install faiss-cpu tevatron qwen-omni-utils
# Note: FAISS search requires GPU for embedding inference
```

## Quick Start

### 1. Install Dependencies

```bash
# Common dependencies
pip install datasets tqdm

# For BM25 search
pip install pyserini tevatron qwen-omni-utils
conda install -c conda-forge openjdk=21

# Or for FAISS
pip install faiss-cpu tevatron qwen-omni-utils
```

### 2. Run Tests

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

### 3. Use BrowseComp Search as a Tool

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
