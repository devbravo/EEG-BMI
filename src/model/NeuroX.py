import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module): 
  def __init__(self, d_in, d_out, ctx_len, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
    
    self.d_out = d_out 
    self.num_heads = num_heads 
    self.head_dim = d_out // num_heads
    
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out) 
    self.dropout = nn.Dropout(dropout) 
    
  def forward(self, x): 
    b, num_tokens, d_in = x.shape 
    
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x) 
    
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
    
    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)
    
    attn_scores = queries @ keys.transpose(2, 3)
    
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights) 
    
    context_vec = (attn_weights @ values).transpose(1, 2) 
    
    context_vec = context_vec.reshape(b, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec) 
    
    return context_vec
  

class LayerNorm(nn.Module):
  def __init__(self, emb_dim, ew_affine=True):
    super().__init__() 
    self.eps = 1e-5 
    self.ew_affine = ew_affine
    
    if self.ew_affine:
      self.scale = nn.Parameter(torch.ones(emb_dim))
      self.shift = nn.Parameter(torch.zeros(emb_dim))
    else: 
      self.gamma = None 
      self.beta = None 
    
  def forward(self, x): 
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    
    if self.ew_affine:
      return self.scale * norm_x + self.shift
    
    return norm_x
  
class GELU(nn.Module): 
  def __init__(self):
    super().__init__() 
    
  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
      torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
      (x + 0.044715 * torch.pow(x, 3))
    ))

class FeedForward(nn.Module): 
  def __init__(self, cfg): 
    super().__init__() 
    self.layers = nn.Sequential(
      nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # * 4 to increase model capacity
      GELU(), 
      nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
    )
    
  def forward(self, x): 
    return self.layers(x) 
  
class Encoder(nn.Module):
  def __init__(self, cfg): 
    super().__init__() 
    self.att = MultiHeadAttention(
      d_in = cfg["emb_dim"],
      d_out = cfg["emb_dim"],
      ctx_len = cfg["ctx_len"],
      num_heads = cfg["n_heads"],
      dropout = cfg["drop_rate"], 
      qkv_bias= cfg["qkv_bias"])
    self.feed_forward = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.drop_resid = nn.Dropout(cfg["drop_rate"])
    
  def forward(self, x): 
    shortcut = x 
    x = self.norm1(x)
    x = self.att(x) 
    x = self.drop_resid(x) 
    x = x + shortcut
    
    shortcut = x 
    x = self.norm2(x)
    x = self.feed_forward(x)
    x = self.drop_resid(x) 
    x = x + shortcut 
    
    return x 
  
class NeuroX(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.input_proj = nn.Linear(cfg["input_dim"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["ctx_len"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])
    
    self.encoder_blocks = nn.Sequential(
      *[Encoder(cfg) for _ in range(cfg["n_layers"])]
    )
    
    self.final_norm = LayerNorm(cfg["emb_dim"])
    self.out_proj = nn.Linear(cfg["emb_dim"], cfg["num_classes"]) if cfg["num_classes"] else None
    
  def forward(self, x):
    batch_size, seq_len, _ = x.shape
    x = self.input_proj(x)
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
    x = x + pos_embeds
    x = self.drop_emb(x)
    x = self.encoder_blocks(x)
    x = self.final_norm(x)
    
    if self.out_proj:
      logits = self.out_proj(x.mean(dim=1))
      return logits
    
    return x

# Example configuration dictionary
cfg = {
  "input_dim": 128,  # Dimension of EEG features
  "emb_dim": 256,    # Embedding dimension
  "ctx_len": 512,    # Context length
  "n_heads": 8,      # Number of attention heads
  "drop_rate": 0.1,  # Dropout rate
  "n_layers": 6,     # Number of transformer layers
  "qkv_bias": True,  # Whether to use bias in QKV linear layers
  "num_classes": 5,  # Number of classes for classification
}

# Instantiate the model
model = NeuroX(cfg)