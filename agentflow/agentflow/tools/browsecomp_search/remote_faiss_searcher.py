"""
Remote FAISS Searcher — uses a shared vLLM embedding server instead of loading
the embedding model locally. This enables multiple eval workers to share one
GPU for embeddings.

Usage:
    # Start embedding server (once):
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-8B \
        --port 8002 --task embed \
        --override-pooler-config '{"pooling_type": "LAST", "normalize": true}' \
        --gpu-memory-utilization 0.3

    # Set env var:
    export EMBEDDING_API_BASE=http://localhost:8002/v1

    # The BrowseComp search tool will auto-detect and use the remote searcher.
"""

import os
import glob
import pickle
import logging
from itertools import chain
from typing import Dict, List, Any, Optional

import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FaissFlatSearcher:
    """Simple FAISS flat (brute-force) index wrapper."""

    def __init__(self, init_reps: np.ndarray):
        self.index = faiss.IndexFlatIP(init_reps.shape[1])

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)


class RemoteFaissSearcher:
    """
    FAISS searcher that uses a remote vLLM embedding server for query encoding.
    Loads the FAISS index and corpus locally, but calls the embedding API
    for query vectorization instead of loading the model in-process.
    """

    def __init__(self, args):
        self.args = args
        self.retriever = None
        self.lookup = None
        self.docid_to_text = None

        # Embedding API config
        self.embedding_api_base = os.environ.get(
            "EMBEDDING_API_BASE", "http://localhost:8002/v1"
        )
        self.embedding_model = getattr(args, "model_name", "Qwen/Qwen3-Embedding-8B")
        self.task_prefix = getattr(
            args, "task_prefix",
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
        )

        logger.info("Initializing Remote FAISS searcher...")
        logger.info(f"Embedding API: {self.embedding_api_base}")
        logger.info(f"Embedding model: {self.embedding_model}")

        self._load_faiss_index()
        self._load_dataset()

        # Verify embedding server is reachable
        self._verify_server()

        logger.info("Remote FAISS searcher initialized successfully")

    def _verify_server(self):
        """Check that the embedding server is reachable."""
        try:
            resp = requests.get(f"{self.embedding_api_base}/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            model_ids = [m["id"] for m in models]
            print(f"[RemoteFAISS] Embedding server OK, models: {model_ids}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach embedding server at {self.embedding_api_base}: {e}\n"
                f"Start it with:\n"
                f"  CUDA_VISIBLE_DEVICES=0 vllm serve {self.embedding_model} "
                f"--port 8002 --task embed "
                f"--override-pooler-config '{{\"pooling_type\": \"LAST\", \"normalize\": true}}'"
            )

    def _load_faiss_index(self) -> None:
        """Load pre-built FAISS index shards from disk."""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu or faiss-gpu is required")

        def pickle_load(path):
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup

        index_path = self.args.index_path
        # Auto-append glob pattern if directory
        if os.path.isdir(index_path):
            index_path = os.path.join(index_path, "corpus.shard*.pkl")

        index_files = sorted(glob.glob(index_path))
        if not index_files:
            raise ValueError(f"No files found matching pattern: {index_path}")

        logger.info(f"Loading {len(index_files)} index shards...")

        # Load first shard
        p_reps_0, p_lookup_0 = pickle_load(index_files[0])
        self.retriever = FaissFlatSearcher(p_reps_0)

        # Load remaining shards
        shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc="Loading shards into index", total=len(index_files))

        self.lookup = []
        for p_reps, p_lookup in shards:
            self.retriever.add(p_reps)
            self.lookup += p_lookup

        # Keep FAISS index on CPU (no GPU needed since we don't load the model)
        logger.info(f"FAISS index loaded: {self.retriever.index.ntotal} vectors")

    def _load_dataset(self) -> None:
        """Load document texts for result display."""
        dataset_name = getattr(self.args, "dataset_name", "Tevatron/browsecomp-plus-corpus")
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, split="train")
            self.docid_to_text = {
                str(row["docid"]): row["text"] for row in dataset
            }
            logger.info(f"Loaded {len(self.docid_to_text)} documents from {dataset_name}")
        except Exception as e:
            logger.warning(f"Could not load dataset {dataset_name}: {e}")
            self.docid_to_text = {}

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query via the remote embedding API."""
        prefixed_query = self.task_prefix + query

        resp = requests.post(
            f"{self.embedding_api_base}/embeddings",
            json={
                "model": self.embedding_model,
                "input": prefixed_query,
                "encoding_format": "float",
            },
            timeout=30,
        )
        resp.raise_for_status()

        data = resp.json()
        embedding = data["data"][0]["embedding"]
        return np.array([embedding], dtype=np.float32)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search the FAISS index using remote embeddings."""
        if self.retriever is None or self.lookup is None:
            raise RuntimeError("Searcher not properly initialized")

        q_reps = self._encode_query(query)
        all_scores, psg_indices = self.retriever.search(q_reps, k)

        results = []
        for score, index in zip(all_scores[0], psg_indices[0]):
            passage_id = self.lookup[index]
            passage_text = self.docid_to_text.get(passage_id, "Text not found")
            results.append({
                "docid": passage_id,
                "score": float(score),
                "text": passage_text,
            })

        return results

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        """Retrieve full document by ID."""
        if not self.docid_to_text:
            raise RuntimeError("Dataset not loaded")

        text = self.docid_to_text.get(docid)
        if text is None:
            return None

        return {"docid": docid, "text": text}

    @property
    def search_type(self) -> str:
        return "RemoteFAISS"
