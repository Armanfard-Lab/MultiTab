import torch.nn as nn
import torch.nn.functional as F
from models.layers import FeatureEmbedding, MultiLayerPerceptron as MLP
   
class ST_MLP(nn.Module):
  def __init__(self, feature_dims, output_dim, hidden_dims=[256], dropout=0.0, embed_dim=0):
    super(ST_MLP, self).__init__()
    feature_dims_list = [v for v in feature_dims.values()]
    self.embedding = FeatureEmbedding(feature_dims_list, embed_dim) if embed_dim > 0 else None
    input_dim = len(feature_dims_list) * (embed_dim if embed_dim else 1)

    self.mlp_layer = MLP(input_dim, hidden_dims, dropout, batch_norm=True)
    self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

  def forward(self, x):
    if self.embedding:
      x = self.embedding(x).flatten(start_dim=1)
    x = self.mlp_layer(x)
    out = self.output_layer(x)
    return out

class MT_MLP(nn.Module):
  def __init__(self, tasks, task_out_dim, feature_dims, shared_hidden_dims=[256], task_hidden_dims=[64], dropout=0.0, embed_dim=0):
      super(MT_MLP, self).__init__()
      feature_dims_list = [v for v in feature_dims.values()]
      self.embedding = FeatureEmbedding(feature_dims_list, embed_dim) if embed_dim > 0 else None
      input_dim = len(feature_dims_list) * (embed_dim if embed_dim else 1)

      self.tasks = tasks
      self.shared_layer = MLP(input_dim, shared_hidden_dims, dropout, batch_norm=True)
      heads = []
      for t in self.tasks:
        if len(task_hidden_dims) == 0:
          head = nn.Linear(shared_hidden_dims[-1], task_out_dim[t])
        else:
          head = nn.Sequential(MLP(shared_hidden_dims[-1], task_hidden_dims, dropout, batch_norm=True),
                               nn.Linear(task_hidden_dims[-1], task_out_dim[t]))
        heads.append(head)
      self.heads = nn.ModuleList(heads)

      # Initialize all linear and embedding layers with mean 0 and std 0.01
      def init_weights(m):
          if isinstance(m, (nn.Linear, nn.Embedding)):
              nn.init.normal_(m.weight, mean=0.0, std=0.01)
              if hasattr(m, 'bias') and m.bias is not None:
                  nn.init.constant_(m.bias, 0.0)
      self.apply(init_weights)

  def forward(self, x):
      out = {}
      if self.embedding:
        x = self.embedding(x).flatten(start_dim=1)
      x = self.shared_layer(x)
      for i, task in enumerate(self.tasks):
        out[task] = self.heads[i](x)
      return out