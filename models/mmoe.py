import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import FeatureEmbedding, Expert, Gate, TaskTower

class MMoE(nn.Module):
    def __init__(self, tasks, feature_dims, num_experts, expert_hidden_dims, tower_hidden_dims, output_dims, dropout=0.0, embed_dim=0):
        super(MMoE, self).__init__()
        self.tasks = tasks
        feature_dims_list = [v for v in feature_dims.values()]
        self.embedding = FeatureEmbedding(feature_dims_list, embed_dim) if embed_dim > 0 else None
        input_dim = len(feature_dims_list) * (embed_dim if embed_dim else 1)

        self.experts = nn.ModuleList([Expert(input_dim, expert_hidden_dims, dropout) for _ in range(num_experts)])
        self.gates = nn.ModuleDict({task: Gate(input_dim, num_experts) for task in self.tasks})
        self.towers = nn.ModuleDict({task: TaskTower(expert_hidden_dims[-1], output_dims[task], tower_hidden_dims, dropout, batch_norm=True) for task in self.tasks})

        # Initialize all linear and embedding layers with mean 0 and std 0.01
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.apply(init_weights)

    def forward(self, x):
        if self.embedding:
            x = self.embedding(x).flatten(start_dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        task_outputs = {}
        for task in self.tasks:
            gate_weights = self.gates[task](x).unsqueeze(2)
            weighted_experts = (expert_outputs * gate_weights).sum(dim=1)
            task_output = self.towers[task](weighted_experts)
            task_outputs[task] = task_output
        return task_outputs
