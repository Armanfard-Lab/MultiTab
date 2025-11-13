import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import FeatureEmbedding, TaskTower

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, ff_hid_dim, ff_dropout, att_dropout, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_dim, num_heads, att_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, ff_hid_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_hid_dim, emb_dim)
        )
        self.dropout = nn.Dropout(ff_dropout)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, input):
        x = self.norm1(input)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x) + input
        
        y = self.mlp(self.norm2(x))
        return x + y
    
class FTT(nn.Module):
    def __init__(self, task_out_dim, feature_dims, tower_hidden_dims, n_blocks=3, embed_dim=32, ff_hid_dim=64, n_heads=4, ff_dropout=0, att_dropout=0):
        super().__init__()
        feature_dims_list = [1] + [v for v in feature_dims.values()]
        self.tok_emb = FeatureEmbedding(feature_dims_list, embed_dim)
        self.blocks = nn.ModuleList([EncoderLayer(embed_dim, ff_hid_dim, ff_dropout, att_dropout, n_heads) for _ in range(n_blocks)])
        self.dropout = nn.Dropout(ff_dropout)

        self.tower = TaskTower(embed_dim, task_out_dim, tower_hidden_dims, ff_dropout, batch_norm=False)

        # Initialize all linear and embedding layers with mean 0 and std 0.01
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.apply(init_weights)

    def forward(self, x):
        B, _ = x.shape
        task_token = torch.ones(B, 1).to(x.device)
        
        tokens = torch.cat((task_token, x), dim=1)
        tokens = self.tok_emb(tokens)     
        x = self.dropout(tokens)

        for block in self.blocks:
            x = block(x)

        out = self.tower(x[:, 0])
        return out