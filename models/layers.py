import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional
from rotary_embedding_torch import RotaryEmbedding

class RotaryRowWiseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = RotaryEmbedding(dim=self.head_dim)

    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        batch_size = x.shape[1]
        seq_len = x.shape[0]

        q = self.q_proj(x)  # (seq_len, batch, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to (batch, heads, seq_len, head_dim)
        def reshape_proj(t):
            t = t.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
            t = t.view(batch_size, seq_len, self.num_heads, self.head_dim)
            return t.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)

        q, k, v = map(reshape_proj, (q, k, v))

        # apply RoPE
        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        # scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (batch, heads, seq_len, head_dim)

        # combine heads
        out = out.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, heads, head_dim)
        out = out.view(batch_size, seq_len, self.embed_dim)
        out = out.permute(1, 0, 2)  # (seq_len, batch, embed_dim)

        return self.out_proj(out)


class FeatureEmbedding(nn.Module):
    def __init__(self, feature_dims, embedding_size):
        super(FeatureEmbedding, self).__init__()
        self.num_features = len(feature_dims)
        self.feature_dims = feature_dims
        self.embedding_size = embedding_size
        
        # Define linear transformations for each feature
        self.embedding_layers = nn.ModuleList()
        for i in range(self.num_features):
            if self.feature_dims[i] == 1:
                self.embedding_layers.append(nn.Linear(1, embedding_size))
            else:
                self.embedding_layers.append(nn.Embedding(self.feature_dims[i], embedding_size))
        
    def forward(self, x):
        # Embed each feature individually
        embedded_features = []
        for i in range(self.num_features):
            feature = x[:, i].long() if self.feature_dims[i] > 1 else x[:, i:i+1]
            embedded_feature = self.embedding_layers[i](feature)  # Embedding each feature separately
            embedded_features.append(embedded_feature)
        
        # Concatenate the embedded features along the feature dimension
        embedded_features = torch.stack(embedded_features, dim=1)
        return embedded_features

class MultiLayerPerceptron(nn.Module):
  def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.0, batch_norm=False):
    super(MultiLayerPerceptron, self).__init__()
    layers = []
    for hidden_size in hidden_sizes:
        layer = [nn.Linear(input_size, hidden_size)]
        if batch_norm:
            layer.append(nn.BatchNorm1d(hidden_size))
        layer.extend([nn.ReLU(), nn.Dropout(p=dropout)])
        layers.append(nn.Sequential(*layer))
        input_size = hidden_size
    self.mlp = nn.Sequential(*layers)

  def forward(self, x):
    return self.mlp(x)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super(Expert, self).__init__()
        self.mlp = MultiLayerPerceptron(input_dim, hidden_dims, dropout, batch_norm=True) # since only MLP models use this
    
    def forward(self, x):
        return self.mlp(x)
    
class Gate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Gate, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)
    
class TaskTower(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], dropout=0.0, batch_norm=False):
        super(TaskTower, self).__init__()
        if len(hidden_dims) > 0:
            self.tower = nn.Sequential(MultiLayerPerceptron(input_dim, hidden_dims, dropout, batch_norm),
                                       nn.Linear(hidden_dims[-1], output_dim))
        else:
            self.tower = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.tower(x)