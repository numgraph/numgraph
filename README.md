<div align="center">

  <img src="Capture2.PNG" alt="numgraph logo" width="800"/>
  
  <p><b>Numerical Core for Learnable Graphs</b></p>

 <p>
    <a href="#-overview">Overview</a> •
    <a href="#-core-architecture">Architecture</a> •
    <a href="#-installation">Installation</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-c-backend-internals">C++ Core</a> •
    <a href="#-python-bindings">Bindings</a> •
    <a href="#-api-reference">API Reference</a> •
    <a href="#-benchmarks">Benchmarks</a> •
    <a href="#-contributing">Contributing</a> •
    <a href="#-license">License</a>
  </p>

  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Build-Passing-success.svg" alt="Build Status">

</div>

numgraph is a high-performance numerical computing library designed specifically for graph-structured data where nodes, edges, and topologies are fully learnable during training loops.Traditional graph neural network frameworks rely on static, human-annotated graph structures that introduce heavy memory overhead and indexing bottlenecks. numgraph solves this by introducing LGraph—a dynamic graph engine powered by a tightly optimized C++17 core with zero-copy Pybind11 bindings for modern tensor runtimes like PyTorch.

##1. Project Overview & Core PhilosophyIn real-world applications (fraud detection, recommendation systems, molecular modeling, and financial flows), physical graph connections are often missing, noisy, or intentionally manipulated. numgraph shifts the paradigm from static message passing to learnable topological evolution.Key ObjectivesLearnable Topology Support: Learn task-optimal network topologies on the fly directly from raw features or unstructured data using low-rank structural decomposition ($E_1 \cdot E_2^T$).Blazing Fast Execution: Bypass Python interpreter overhead for intensive neighborhood aggregation and node-state updates using parallel C++ kernels.Memory-Efficient Sparsification: Control neighborhood density and memory footprints using dynamic $O(N \cdot k)$ top_k sparse filters.Modular Extensibility: Clear isolation of memory layouts, sparse graph storage formats (CSR/CSC), and custom compute kernels.

##2. Core ArchitectureThe system is split into two primary layers:┌────────────────────────────────────────────────────────┐
│                   Python Frontend                      │
│   • numgraph.LGraph (nn.Module Wrapper)                │
│   • PyTorch Autograd Engine Integration                │
└───────────────────────────┬────────────────────────────┘
                            │ Zero-Copy Tensors (pybind11)
┌───────────────────────────▼────────────────────────────┐
│                    C++ Core Engine                     │
│   • Low-Rank Tensor Products (E1 • E2ᵀ)                │
│   • Parallel OpenMP Top-K Aggregation Kernels          │
│   • CSR / CSC Sparse Representation Managers           │
└────────────────────────────────────────────────────────┘
The C++ Engine (csrc/): Manages raw memory buffers, multi-threaded CPU kernels (via OpenMP), low-rank topology calculations, and sparse graph layout transformations.The Python Frontend (numgraph/): Exposes idiomatic PyTorch module wrappers (nn.Module), autograd functions (torch.autograd.Function), and scalable model assembly blocks.

##3. Why LGraph? Core Qualities & CapabilitiesThe flagship operator in numgraph is LGraph (Learnable Graph). Unlike traditional GNN layers that require an explicit edge_index input, LGraph decouples your model from rigid database connections.[N Nodes]
   │
   ├─► E1 (N x embed_dim) ──┐
   │                        ├──► Outer Product (E1 • E2ᵀ) ──► Low-Rank Scores (N x N)
   └─► E2 (N x embed_dim) ──┘                                         │
                                                                      ▼
                                                            top_k Sparsify & Softmax
                                                                      │
                                                                      ▼
                                                            Sparse Dynamic Graph (N x k)
                                                                      │
                                                                      ▼
                                                            Message Passing Output
Key Qualities of LGraph:No-Graph-Required Interface: Accepts raw feature matrices $\mathbf{X} \in \mathbb{R}^{N \times F}$ directly. It synthesizes an interaction graph from scratch for unstructured tabular data, point clouds, or multi-modal vectors.Low-Rank Topology Memory Scaling: Computes global pairwise similarities using factorized parameters $E_1, E_2 \in \mathbb{R}^{N \times d}$. This reduces structural parameter memory from quadratic $O(N^2)$ to linear $O(N \cdot d)$, allowing numgraph to scale to massive node counts ($N$).Dynamic top_k Noise Pruning: Automatically prunes weak, noisy, or deceptive connections, retaining only the top $k$ strongest incoming connections per node.Oversmoothing Prevention: Features built-in residual connections, pre-layer normalization, and multi-head attention to allow deep stacking without feature collapse or gradient degradation

##4. Installation & SetupPrerequisites
A modern C++17 compliant compiler (g++ >= 9.0, clang >= 10, or MSVC).Python 3.8 or higher.PyTorch (version 2.0.0 or greater).Source InstallationClone the repository and install it in editable mode:Bashgit 
```bash
clone https://github.com/numgraph/numgraph.git
cd numgraph
pip install -e . --verbose
```


##5. Quick Start GuideHere is how to initialize an LGraph layer and execute a forward/backward pass within a standard PyTorch training loop:Pythonimport torch
import numgraph as ng

```python
# Define master node dimension and input features
num_nodes = 1000
in_features = 64
out_features = 32

# 1. Initialize sample node feature tensor (No edge_index needed!)
node_features = torch.randn(num_nodes, in_features, requires_grad=True)

# 2. Initialize the Learnable LGraph Operator
# - embed_dim: Rank d of structural identity matrices E1, E2
# - top_k: Number of active sparse neighbors retained per node
layer = ng.LGraph(
    num_nodes=num_nodes,
    in_features=in_features,
    out_features=out_features,
    embed_dim=32,
    top_k=10,
    num_heads=4
)

# 3. Forward Pass (Dynamically constructs topology & aggregates features)
output = layer(node_features)
print("Output tensor shape:", output.shape)  # torch.Size([1000, 32])

# 4. Compute Loss and Backpropagate Gradients
loss = output.sum()
loss.backward()

print("Node features gradient norm:", node_features.grad.norm().item())
print("Learned E1 topology parameter gradient norm:", layer.E1.grad.norm().item())
Stacking Deep LGraph NetworksLGraph blocks can be stacked seamlessly to capture higher-order relational dependencies:Pythonimport torch.nn as nn
import numgraph as ng

class EnterpriseGMLModel(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = ng.LGraph(num_nodes, in_dim, hidden_dim, embed_dim=32, top_k=15, num_heads=4)
        self.layer2 = ng.LGraph(num_nodes, hidden_dim, hidden_dim, embed_dim=32, top_k=10, num_heads=4)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.classifier(x)
```
##6. C++ Backend InternalsThe C++ core in csrc/ optimizes memory access patterns and minimizes cache misses during sparse aggregation.Sparse Storage & Neighborhood AggregationC++
``` cpp
#include <torch/extension.h>
#include <omp.h>
#include <vector>

// High-performance CPU multithreaded neighborhood aggregation routine
torch::Tensor aggregate_topk_cpu(
    const torch::Tensor& value_features,  // [H, N, head_dim]
    const torch::Tensor& topk_indices,    // [H, N, k]
    const torch::Tensor& attn_weights     // [H, N, k]
) {
    auto H = value_features.size(0);
    auto N = value_features.size(1);
    auto k = topk_indices.size(2);
    auto head_dim = value_features.size(2);

    auto output = torch::zeros({N, H * head_dim}, value_features.options());

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < N; ++i) {
            for (int neighbor_idx = 0; neighbor_idx < k; ++neighbor_idx) {
                int src_node = topk_indices[h][i][neighbor_idx].item<int>();
                float weight = attn_weights[h][i][neighbor_idx].item<float>();

                for (int d = 0; d < head_dim; ++d) {
                    output[i][h * head_dim + d] += weight * value_features[h][src_node][d];
                }
            }
        }
    }
    return output;
}
```
##7. Python Bindings & PyTorch Integration
Zero-Copy Memory Overhead: Native PyTorch tensors pass directly into C++ pointer addresses using pybind11::array_t and Torch C++ API wrappers.Autograd Registration: Forward computation passes through torch.autograd.Function, enabling seamless gradient calculation back to $E_1, E_2$, weights, and biases.Device Agnosticism: Core execution layers automatically dispatch workloads to CPU thread pools via OpenMP or stream executors on matching hardware targets.

##8. Development Roadmap
[x] Core C++ architecture and project setup.[x] Pybind11 bindings and basic setup script configuration.[x] Native LGraph low-rank structural topology layer ($E_1 \cdot E_2^T$).[x] Multi-head top-k sparse aggregation routines.[ ] Custom CUDA/GPU fused kernels for top_k gathering in csrc/.[ ] Dynamic batching offsets for multi-graph batch processing.[ ] Distributed multi-node graph partitioning utilities.

##9. Contributing
We welcome contributions! Whether optimizing C++ routines, refining LGraph topology operations, or expanding documentation:Fork the repository.Create your feature branch (git checkout -b feature/AmazingFeature).Commit your changes (git commit -m 'Add some AmazingFeature').Push to the branch (git push origin feature/AmazingFeature).Open a Pull Request.Please review CONTRIBUTING.md for code style and test suite guidelines.
