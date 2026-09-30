# numgraph/l_graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LGraph(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        out_features: int,
        emb_dim: int = 16,
        top_k: int = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.top_k = top_k if top_k is not None else 0

        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(emb_dim, num_nodes) * 0.1)

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def _compute_adaptive_adj(self) -> torch.Tensor:
        raw_adj = F.relu(torch.mm(self.node_emb1, self.node_emb2))

        if self.top_k > 0 and self.top_k < self.num_nodes:
            topk_val, topk_idx = torch.topk(raw_adj, self.top_k, dim=-1)
            mask = torch.zeros_like(raw_adj).scatter_(-1, topk_idx, 1.0)
            raw_adj = raw_adj * mask

        adj = F.softmax(raw_adj + 1e-6, dim=-1)
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adj = self._compute_adaptive_adj()
        support = torch.matmul(x, self.weight)

        if x.dim() == 2:
            return torch.matmul(adj, support)
        elif x.dim() == 3:
            return torch.matmul(adj.unsqueeze(0), support)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

    def __repr__(self) -> str:
        return f"LGraph({self.num_nodes} , {self.in_features} , {self.out_features} , {self.top_k})"