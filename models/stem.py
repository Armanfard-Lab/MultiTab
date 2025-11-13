import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import FeatureEmbedding, Expert, Gate, TaskTower

class STEMLayer(nn.Module):
    def __init__(self, input_dim, num_shared_experts, num_task_experts, num_tasks, expert_hidden_dims, dropout):
        super(STEMLayer, self).__init__()
        self.num_tasks = num_tasks

        self.shared_experts = nn.ModuleList([Expert(input_dim, expert_hidden_dims, dropout) for _ in range(num_shared_experts)])
        self.shared_gate = Gate(input_dim, num_shared_experts + num_tasks*num_task_experts)
        self.task_experts = nn.ModuleList([nn.ModuleList([Expert(input_dim, expert_hidden_dims, dropout) for _ in range(num_task_experts)]) for _ in range(num_tasks)])
        self.task_gates = nn.ModuleList([Gate(input_dim, num_shared_experts + num_tasks*num_task_experts) for _ in range(num_tasks)])

    def forward(self, x):
        """
        x: list, len(x)==num_tasks+1
        """
        # Get the weight values for every gate
        gate_values = [gate(x[i]) for i, gate in enumerate(self.task_gates + [self.shared_gate])]
        
        # Get the outputs of each task expert and then the shared expert
        expert_outputs = []
        for i in range(self.num_tasks):
            expert_outputs.append(torch.stack([task_experts(x[i]) for task_experts in self.task_experts[i]], dim=1))
        expert_outputs.append(torch.stack([shared_expert(x[-1]) for shared_expert in self.shared_experts], dim=1))
        
        # Compute the output of each task gate and then the shared gate
        gated_outputs = []
        for i in range(self.num_tasks):
            expert_outputs_stop_grad = []
            for j in range(len(expert_outputs)):
                if j == i or j == -1:
                    expert_outputs_stop_grad.append(expert_outputs[j])
                else:
                    # Apply stop gradient to embeddings from other tasks
                    expert_outputs_stop_grad.append(expert_outputs[j].detach())
            selected_matrix = torch.cat(expert_outputs_stop_grad, dim=1)
            gated_outputs.append(torch.sum(selected_matrix * gate_values[i].unsqueeze(-1), dim=1))
        shared_selected_matrix = torch.cat(expert_outputs, dim=1)
        gated_outputs.append(torch.sum(shared_selected_matrix * gate_values[-1].unsqueeze(-1), dim=1))
        return gated_outputs

class STEM(nn.Module):
    def __init__(self, tasks, num_layers, num_shared_experts, num_task_experts, expert_hidden_dims, tower_hidden_dims, task_output_dims, feature_dims, embed_dim, dropout):
        super(STEM, self).__init__()
        self.tasks = tasks
        feature_dims_list = [v for v in feature_dims.values()]
        self.embeddings = nn.ModuleList([FeatureEmbedding(feature_dims_list, embed_dim) for _ in range(len(self.tasks)+1)])
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            expert_input_dim = len(feature_dims)*embed_dim if i==0 else expert_hidden_dims[-1]
            self.layers.append(STEMLayer(expert_input_dim, num_shared_experts, num_task_experts, len(tasks), expert_hidden_dims, dropout))
        self.towers = nn.ModuleDict({task: TaskTower(expert_hidden_dims[-1]*2, task_output_dims[task], tower_hidden_dims, dropout, batch_norm=True) for task in tasks})

        # Initialize all linear and embedding layers with mean 0 and std 0.01
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.apply(init_weights)

    def forward(self, x):
        y = [self.embeddings[i](x).flatten(start_dim=1) for i in range(len(self.tasks)+1)] # list of inputs for each set of task experts and shared expert
        for layer in self.layers:
            y = layer(y)
        out = {}
        for i, task in enumerate(self.tasks):
            out[task] = self.towers[task](torch.cat((y[i],y[-1]), dim=1))
        return out
