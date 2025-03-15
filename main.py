''' Imported Libraries '''

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset
import math

''' Config '''

GPT2_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_blocks": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

GPT2_CONFIG["ff_hidden_size"] = 4 * GPT2_CONFIG["emb_dim"]

print(f"device:\t{GPT2_CONFIG['device']}")

''' tokenizer '''

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "My name is Yash"
enc_inp = tokenizer.encode(text, return_tensors='pt').to(GPT2_CONFIG["device"])
print(f"input tokens shape:\t{enc_inp.shape}")

''' Embedding '''

class Embedding(nn.Module):
    def __init__(self, vocab_size=GPT2_CONFIG["vocab_size"], emb_dim=GPT2_CONFIG["emb_dim"], device=GPT2_CONFIG["device"]):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, device=device)

    def forward(self, x):
        out = self.emb(x)

        print(f"embedded tokens shape:\t{out.shape}")
        return out

''' RoPE (Rotary Positional Encoding) '''

class RoPE(nn.Module):
    def __init__(self, max_seq_len=GPT2_CONFIG["context_length"], emb_dim=GPT2_CONFIG["emb_dim"], device=GPT2_CONFIG["device"]):
        super().__init__()

        self.max_seq_len = max_seq_len 
        self.emb_dim = emb_dim
        self.device = device

        self.theta = 10000 ** (-2 * (torch.arange(0, emb_dim, 2, dtype=torch.float, device=device))/emb_dim)
        self.position = torch.arange(0, max_seq_len, dtype=torch.float, device=device).unsqueeze(1)

        freqs = self.position * self.theta

        cos_terms = torch.cos(freqs)
        sin_terms = torch.sin(freqs)

        self.register_buffer('cos_terms', torch.repeat_interleave(cos_terms, 2, dim=-1))
        self.register_buffer('sin_terms', torch.repeat_interleave(sin_terms, 2, dim=-1))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        cos = self.cos_terms[:seq_len]
        sin = self.sin_terms[:seq_len]

        new_x = torch.empty_like(x)
        new_x[..., 0::2] = -x[..., 1::2]
        new_x[..., 1::2] = x[..., 0::2]

        out = x * cos.unsqueeze(0) + new_x * sin.unsqueeze(0)

        print(f"RoPE shape: {out.shape}")
        assert x.shape == out.shape, "input embedding shape does not match RoPE shape"
        return out

''' RMSNorm (Root Mean Square Normalization) '''

class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6, emb_dim=GPT2_CONFIG["emb_dim"], device=GPT2_CONFIG["device"]):
        super().__init__()

        self.eps = eps
        self.emb_dim = emb_dim

        self.gamma = torch.ones([self.emb_dim], requires_grad=True, device=device)

    def forward(self, x):
        rms = torch.sqrt(self.eps + torch.mean(x ** 2))
        scaled = (x / rms) * self.gamma

        print(f"RMSNorm shape: {scaled.shape}")
        assert x.shape == scaled.shape, "input shape does not match RMSNorm shape"
        return scaled

''' SwiGLU Activation Function '''

class SwiGLU(nn.Module):
    def __init__(self, emb_dim=GPT2_CONFIG["emb_dim"], device=GPT2_CONFIG["device"]):
        super().__init__()

        h = int(8 * emb_dim / 3)

        self.w = nn.Linear(4 * emb_dim, h, device=device)
        self.v = nn.Linear(4 * emb_dim, h, device=device)
        self.down_proj = nn.Linear(h, 4 * emb_dim, device=device)

    def forward(self, x, device=GPT2_CONFIG["device"]):
        b, t, c = x.shape

        wx = self.w(x)
        vx = self.v(x)

        swished = vx * torch.sigmoid(vx)
        out = self.down_proj(wx * swished)

        print(f"SwiGLU shape: {out.shape}")
        assert x.shape == out.shape, "input shape does not match SwiGLU shape"
        return out
        
''' Masked Attention Head '''

class Head(nn.Module):
    def __init__(self, head_size, emb_dim=GPT2_CONFIG["emb_dim"], qkv_bias=GPT2_CONFIG["qkv_bias"], device=GPT2_CONFIG["device"]):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.head_size = head_size

        self.rope = RoPE(emb_dim=head_size)

        self.qw = nn.Linear(emb_dim, head_size, bias=qkv_bias, device=device)
        self.kw = nn.Linear(emb_dim, head_size, bias=qkv_bias, device=device)
        self.vw = nn.Linear(emb_dim, head_size, bias=qkv_bias, device=device)

    def forward(self, x, device=GPT2_CONFIG["device"]):
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        rq = self.rope(q)
        rk = self.rope(k)

        mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf'), device=device), diagonal=1)

        qk = rq @ rk.transpose(-2, -1)
        scaling = qk * (self.emb_dim ** -0.5)
        add_mask = scaling + mask
        scaled_sm = F.softmax(add_mask, dim=-1)
        qk_v = scaled_sm @ v

        print(f"head shape:\t{qk_v.shape}")
        return qk_v

''' Multi Head Masked Attention '''

class Multi_Head(nn.Module):
    def __init__(self, emb_dim=GPT2_CONFIG["emb_dim"], n_heads=GPT2_CONFIG["n_heads"], device=GPT2_CONFIG["device"]):
        super().__init__()

        self.head_size = emb_dim // n_heads

        self.heads = nn.ModuleList([Head(self.head_size) for _ in range(n_heads)])
        self.lyr = nn.Linear(emb_dim, emb_dim, bias=False, device=device)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.lyr(out)

        assert x.shape == out.shape, "positional encoding input shape does not match multi-head output shape"
        print(f"multi head shape:\t{out.shape}")
        return out

''' Feed Forward Layer '''

class FeedForward(nn.Module):
    def __init__(self, emb_dim=GPT2_CONFIG["emb_dim"], ff_hidden_size=GPT2_CONFIG["ff_hidden_size"], device=GPT2_CONFIG["device"]):
        super().__init__()

        self.lyr_1 = nn.Linear(emb_dim, ff_hidden_size, device=device)
        self.lyr_2 = nn.Linear(ff_hidden_size, emb_dim, device=device)
        # self.gelu = nn.GELU()
        self.swiglu = SwiGLU()

    def forward(self, x):
        out = self.lyr_1(x)
        out = self.swiglu(out)
        out = self.lyr_2(out)

        print(f"feed forward shape:\t{out.shape}")
        return out

''' Transformer Block '''

class Block(nn.Module):
    def __init__(self, emb_dim=GPT2_CONFIG["emb_dim"], drop_rate=GPT2_CONFIG["drop_rate"]):
        super().__init__()

        self.m_head = Multi_Head()
        self.ff = FeedForward()
        self.rms1 = RMSNorm()
        self.rms2 = RMSNorm()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        out = x + self.m_head(self.rms1(x))
        out = out + self.ff(self.rms2(out))

        print(f"block shape:\t{out.shape}")
        return out

''' GPT2 Model Class '''

class GPT2(nn.Module):
    def __init__(self, n_blocks=GPT2_CONFIG["n_blocks"], emb_dim=GPT2_CONFIG["emb_dim"], vocab_size=GPT2_CONFIG["vocab_size"], device=GPT2_CONFIG["device"]):
        super().__init__()

        self.embd = Embedding()

        self.block_list = nn.ModuleList([Block() for _ in range(n_blocks)])
        self.rmsnorm = RMSNorm()
        self.lyr = nn.Linear(emb_dim, vocab_size, device=device)

    def forward(self, x):
        out = self.embd(x)

        for block in self.block_list:
            out = block(out)

        out = self.lyr(self.rmsnorm(out))

        print(f"GPT2 shape:\t{out.shape}")
        return out

gpt2 = GPT2()
gpt2.to(GPT2_CONFIG["device"])
x_gpt2 = gpt2(enc_inp)

gpt2_params = sum(p.numel() for p in gpt2.parameters())
print(f"GPT2 Params:\t{gpt2_params}")
