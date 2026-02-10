#!/usr/bin/env python3
"""
Test script for BrowseComp-Plus search (BM25 and FAISS).

Requirements:
- Common: datasets, tqdm, tevatron, qwen-omni-utils
- BM25: Java JDK 21 + pyserini
- FAISS: faiss-cpu

Usage:
    python test_browsecomp.py --index-type bm25    # Requires Java
    python test_browsecomp.py --index-type faiss   # No Java required
"""
import sys
import os
import argparse
from pathlib import Path

# Add BrowseComp-Plus to path
browsecomp_path = Path(__file__).parent / "BrowseComp-Plus"
sys.path.insert(0, str(browsecomp_path))

from datasets import load_dataset

def test_bm25_search(index_path):
    """Test BM25 search functionality."""
    try:
        from searcher.searchers.bm25_searcher import BM25Searcher
        from argparse import Namespace

        print("Testing BrowseComp-Plus BM25 Searcher...")
        args = Namespace(index_path=index_path)
        searcher = BM25Searcher(args)

        query = "What is the capital of France?"
        print(f"\nQuery: {query}")
        print("=" * 50)

        results = searcher.search(query, k=3)

        for i, result in enumerate(results, 1):
            docid = result.get("docid")
            score = result.get("score", 0)
            text = result.get("text", "")[:200]
            print(f"{i}. DocID: {docid} (Score: {score:.4f})")
            print(f"   {text}...")

        print("\n✓ BM25 Search Test completed!")
        return True
    except ImportError as e:
        print(f"✗ BM25 not available: {e}")
        print("Install: pip install pyserini && conda install -c conda-forge openjdk=21")
        return False
    except Exception as e:
        print(f"✗ BM25 test failed: {e}")
        return False

def test_faiss_search(index_path):
    """Test FAISS search functionality."""
    try:
        from searcher.searchers.faiss_searcher import FaissSearcher
        from argparse import Namespace

        print("Testing BrowseComp-Plus FAISS Searcher...")

        # FAISS needs a glob pattern, not a directory
        # e.g. /path/to/indexes/qwen3-embedding-0.6b/corpus.shard*.pkl
        glob_pattern = os.path.join(index_path, "corpus.shard*.pkl")

        # Get model size from index path
        if "0.6b" in index_path:
            model_name = "Qwen/Qwen3-Embedding-0.6B"
        elif "4b" in index_path:
            model_name = "Qwen/Qwen3-Embedding-4B"
        elif "8b" in index_path:
            model_name = "Qwen/Qwen3-Embedding-8B"
        else:
            model_name = "Qwen/Qwen3-Embedding-0.6B"  # default

        args = Namespace(
            index_path=glob_pattern,
            model_name=model_name,
            normalize=False,
            pooling="eos",
            torch_dtype="float16",
            dataset_name="Tevatron/browsecomp-plus-corpus",
            max_length=8192
        )
        searcher = FaissSearcher(args)

        query = "What is the capital of France?"
        print(f"Model: {model_name}")
        print(f"Query: {query}")
        print("=" * 50)

        results = searcher.search(query, k=3)

        for i, result in enumerate(results, 1):
            docid = result.get("docid")
            score = result.get("score", 0)
            text = result.get("text", "")[:200]
            print(f"{i}. DocID: {docid} (Score: {score:.4f})")
            print(f"   {text}...")

        print("\n✓ FAISS Search Test completed!")
        return True
    except ImportError as e:
        print(f"✗ FAISS not available: {e}")
        print("Install: pip install faiss-cpu")
        return False
    except Exception as e:
        print(f"✗ FAISS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_corpus_loading():
    """Test corpus loading from HuggingFace."""
    print("Testing BrowseComp-Plus corpus loading...")
    try:
        corpus = load_dataset('Tevatron/browsecomp-plus-corpus', split='train')
        print(f"✓ Corpus loaded: {len(corpus):,} documents")
        return True
    except Exception as e:
        print(f"✗ Corpus loading failed: {e}")
        return False

def test_query_loading():
    """Test query loading/decryption."""
    print("Testing BrowseComp-Plus query loading...")
    try:
        dataset = load_dataset('Tevatron/browsecomp-plus', split='test')
        print(f"✓ Queries loaded: {len(dataset):,} queries")
        return True
    except Exception as e:
        print(f"✗ Query loading failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BrowseComp-Plus integration")
    parser.add_argument("--index-type", choices=["bm25", "faiss", "all"], default="all",
                        help="Search index type to test")
    parser.add_argument("--no-search", action="store_true",
                        help="Skip search test, only test dataset loading")
    args = parser.parse_args()

    print("=" * 60)
    print("BrowseComp-Plus Integration Test")
    print("=" * 60)
    print()

    # Setup JAVA_HOME for BM25
    if args.index_type == "bm25" and not args.no_search:
        if "JAVA_HOME" not in os.environ:
            default_java = os.path.expanduser("~/miniconda3")
            if os.path.exists(default_java):
                os.environ["JAVA_HOME"] = default_java
                print(f"Set JAVA_HOME={default_java}")

    # Run tests
    results = []

    print("\n--- Dataset Tests ---")
    results.append(("Corpus", test_corpus_loading()))
    results.append(("Queries", test_query_loading()))

    if not args.no_search:
        if args.index_type == "all":
            # Test BM25
            index_path = str(Path(__file__).parent / "BrowseComp-Plus/indexes/bm25")
            print(f"\n--- BM25 Search Test ---")
            results.append(("BM25 Search", test_bm25_search(index_path)))

            # Test FAISS (use the smallest one - 0.6b)
            index_path = str(Path(__file__).parent / "BrowseComp-Plus/indexes/qwen3-embedding-0.6b")
            print(f"\n--- FAISS Search Test ---")
            results.append(("FAISS Search", test_faiss_search(index_path)))
        elif args.index_type == "bm25":
            index_path = str(Path(__file__).parent / "BrowseComp-Plus/indexes/bm25")
            print(f"\n--- BM25 Search Test ---")
            results.append(("BM25 Search", test_bm25_search(index_path)))
        else:  # faiss
            index_path = str(Path(__file__).parent / "BrowseComp-Plus/indexes/qwen3-embedding-0.6b")
            print(f"\n--- FAISS Search Test ---")
            results.append(("FAISS Search", test_faiss_search(index_path)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")

    all_passed = all(r[1] for r in results)
    print("=" * 60)
    if all_passed:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)
