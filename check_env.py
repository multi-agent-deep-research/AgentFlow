#!/usr/bin/env python3
"""Quick check that all dependencies are working."""
import torch
print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")
import torchvision
print(f"torchvision: {torchvision.__version__}")
try:
    import flash_attn
    print(f"flash-attn: {flash_attn.__version__}")
except ImportError:
    print("flash-attn: NOT INSTALLED (FAISS will use eager attention)")
from transformers import PreTrainedModel
print("transformers: OK")
print(f"GPU available: {torch.cuda.is_available()}")
