import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import TaskTower

class FeatureEmbedding(nn.Module):
    def __init__(self, num_features, feature_dims, embedding_size):
        super(FeatureEmbedding, self).__init__()
        self.num_features = num_features
        self.feature_dims = [v for v in feature_dims.values() if v > 1] # 1 continuous, 2 binary, > 2 categorical
        self.embedding_size = embedding_size
        
        # Define linear transformations for each feature
        self.embedding_layers = nn.ModuleList()
        for i in range(num_features):
            self.embedding_layers.append(nn.Embedding(self.feature_dims[i], embedding_size, sparse=False))
        
    def forward(self, x):
        # Embed each feature individually
        embedded_features = []
        for i in range(self.num_features):
            embedded_feature = self.embedding_layers[i](x[:, i:i+1])  # Embedding each feature separately
            embedded_features.append(embedded_feature)
        
        # Concatenate the embedded features along the feature dimension
        embedded_features = torch.cat(embedded_features, dim=1)
        return embedded_features

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, ff_hid_dim, ff_dropout, att_dropout, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_dim, num_heads, att_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, ff_hid_dim),
            nn.GELU(),
            #nn.Dropout(ff_dropout),
            nn.Linear(ff_hid_dim, emb_dim)
        )
        self.dropout = nn.Dropout(ff_dropout)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x_norm = self.norm1(x)
        attention, _ = self.attention(x_norm, x_norm, x_norm)
        x = attention + self.dropout(x)
        out = self.mlp(self.norm2(x))
        out = out + self.dropout(x)
        return out
    
class TabTransformer(nn.Module):
    def __init__(self, task_out_dim, feature_dims, tower_hidden_dims, n_blocks=3, embed_dim=32, ff_hid_dim=64, n_heads=4, ff_dropout=0, att_dropout=0):
        super().__init__()

        self.num_features_cont = sum(1 for value in feature_dims.values() if value == 1)
        self.num_features_cat = sum(1 for value in feature_dims.values() if value > 1)
        
        self.norm_cont = nn.LayerNorm(self.num_features_cont) if self.num_features_cont > 0 else None
        self.tok_emb = FeatureEmbedding(self.num_features_cat, feature_dims, embed_dim)
        self.blocks = nn.ModuleList([EncoderLayer(embed_dim, ff_hid_dim, ff_dropout, att_dropout, n_heads) for _ in range(n_blocks)])

        self.tower = TaskTower(self.num_features_cat*embed_dim + self.num_features_cont, task_out_dim, tower_hidden_dims, ff_dropout, batch_norm=False)

        # Initialize all linear and embedding layers with mean 0 and std 0.01
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.apply(init_weights)

    def forward(self, x):       
        x_con = self.norm_cont(x['continuous']) if self.num_features_cont > 0 else None
        x_cat = self.tok_emb(x['categorical'])  

        for block in self.blocks:
            x_cat = block(x_cat)

        x_combined = torch.cat((x_cat.flatten(start_dim=1), x_con), dim=1) if x_con is not None else x_cat.flatten(start_dim=1)
        out = self.tower(x_combined)
        return out