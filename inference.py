import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
import time
from typing import Optional
from pathlib import Path
import json
from tqdm import tqdm

from model import Transformer, LLaMA_2_CONFIG

class LLaMA:
    def _sample_top_p(self, probs, p):
        probs_sorted, probs_idx = torch.sort(p, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sorted, dim=-1)
        mask = probs_sum - probs_sorted > p
        probs_sorted[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sorted, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        
        return next_token
