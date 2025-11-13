import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import FeatureEmbedding, TaskTower, RotaryRowWiseAttention

def create_mtl_mask(seq_length, num_task_tokens, mask_mode='TxT', device='cpu'):
    mask = torch.zeros((seq_length, seq_length), device=device)
    if mask_mode in {'FxT', 'FxT_TxT'}:  # Mask task tokens from features
        mask[num_task_tokens:, :num_task_tokens] = 1
    if mask_mode in {'TxT', 'FxT_TxT'}:  # Mask task tokens from other task tokens
        mask[:num_task_tokens, :num_task_tokens] = 1
        mask[torch.arange(num_task_tokens), torch.arange(num_task_tokens)] = 0  # Preserve self-attention
    if mask_mode not in {'FxT', 'TxT', 'FxT_TxT'}:
        raise NotImplementedError(f"Mask mode {mask_mode} not implemented")
    return mask

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, ff_hid_dim, ff_dropout, att_dropout, num_heads, num_task_tokens, mask_mode='none', rope=False):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.mask_mode = mask_mode
        self.rotary = rope

        self.attention = nn.MultiheadAttention(emb_dim, num_heads, dropout=att_dropout, batch_first=True) if not rope else RotaryRowWiseAttention(emb_dim, num_heads)
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
        if self.mask_mode != 'none' and self.training:
            mask = create_mtl_mask(input.shape[1], self.num_task_tokens, self.mask_mode, device=input.device).bool()
        else:
            if self.mask_mode in {'FxT', 'TxT', 'FxT_TxT'}:
                mask = create_mtl_mask(input.shape[1], self.num_task_tokens, self.mask_mode, device=input.device).bool()
            else:
                mask = None
        x = self.norm1(input)
        if self.rotary:
            x = self.attention(x)
        else:
            x, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.dropout(x) + input
        
        y = self.mlp(self.norm2(x))
        return x + y

class RowColTransformer(nn.Module):
    def __init__(self, n_features, embed_dim, ff_hid_dim, ff_dropout, att_dropout, n_heads, n_blocks, num_task_tokens, mask_mode_if, rope=False):
       super().__init__()
       self.n_features = n_features
       self.blocks = nn.ModuleList([
            nn.ModuleList([
                EncoderLayer(embed_dim, ff_hid_dim, ff_dropout, att_dropout, n_heads, num_task_tokens, mask_mode_if, rope=False), # inter-feature
                EncoderLayer(embed_dim * n_features, ff_hid_dim, ff_dropout, att_dropout, n_heads, num_task_tokens, 'none', rope=rope)  # inter-sample
            ]) for _ in range(n_blocks)
        ])
       
    def forward(self, x):
        B = x.shape[0]
        for block in self.blocks:
            x = block[0](x)
            x = x.reshape(1, B, -1) # (1, b, n*d)
            x = block[1](x)
            x = x.squeeze(0)
            x = x.reshape(B, self.n_features, -1) # (b, n, d)
        return x

class ColTransformer(nn.Module):
    def __init__(self, embed_dim, ff_hid_dim, ff_dropout, att_dropout, n_heads, n_blocks, num_task_tokens, mask_mode):
       super().__init__()
       self.blocks = nn.ModuleList([
            nn.ModuleList([
                EncoderLayer(embed_dim, ff_hid_dim, ff_dropout, att_dropout, n_heads, num_task_tokens, mask_mode), # inter-feature
            ]) for _ in range(n_blocks)
        ])
       
    def forward(self, x):
        for block in self.blocks:
            for layer in block:
                x = layer(x)  # Call the forward method of each layer explicitly
        return x

class MTT(nn.Module):
    def __init__(self, tasks, task_out_dim, feature_dims, tower_hid_dims, n_blocks=3, embed_dim=32, ff_hid_dim=64, n_heads=4, ff_dropout=0, att_dropout = 0.05,  mask_mode_if='none', att_type='col', multi_token = True, rope=False):
        super().__init__()
        self.tasks = tasks
        self.multi_token = multi_token

        num_task_tokens = len(tasks) if multi_token else 1
        feature_dims_list = [1]*num_task_tokens + [v for v in feature_dims.values()]

        self.tok_emb = FeatureEmbedding(feature_dims_list, embed_dim)
        # self.blocks = nn.ModuleList([EncoderLayer(embed_dim, ff_hid_dim, dropout, n_heads, len(tasks), mask_mode) for _ in range(n_blocks)])
        if att_type == 'col':
            self.transformer = ColTransformer(embed_dim, ff_hid_dim, ff_dropout, att_dropout, n_heads, n_blocks, num_task_tokens, mask_mode_if)
        elif att_type == 'row_col':
            self.transformer = RowColTransformer(len(feature_dims_list), embed_dim, ff_hid_dim, ff_dropout, att_dropout, n_heads, n_blocks, num_task_tokens, mask_mode_if, rope)
        else:
            raise NotImplementedError(f"Attention type {att_type} not implemented") 
        self.dropout = nn.Dropout(ff_dropout)
        
        self.task_heads = nn.ModuleDict({
            task: TaskTower(embed_dim, task_out_dim[task], tower_hid_dims, ff_dropout, batch_norm=False) for task in self.tasks}
        )

        # Initialize all linear and embedding layers with mean 0 and std 0.01
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.apply(init_weights)

    def forward(self, x):
        out = {}
        B, _ = x.shape
        if self.multi_token:
            task_tokens = torch.ones(B, len(self.tasks)).to(x.device)
        else:
            task_tokens = torch.ones(B, 1).to(x.device)
        
        tokens = torch.cat((task_tokens, x), dim=1)
        tokens = self.tok_emb(tokens)

        x = self.dropout(tokens)

        x = self.transformer(x)
        
        for i, task in enumerate(self.tasks):
            if self.multi_token:
                out[task] = self.task_heads[task](x[:,i])
            else:
                out[task] = self.task_heads[task](x[:,0])
        return out