# tests/test_l_graph.py
import time
import torch
from numgraph import LGraph


def run_tests():
    start_time = time.time()

    lgraph = LGraph(
        num_nodes=5,
        in_features=16,
        out_features=8,
        top_k=3,
    )

    print(lgraph)

    x_2d = torch.randn(5, 16, requires_grad=True)
    out_2d = lgraph(x_2d)
    loss = out_2d.sum()
    loss.backward()

    elapsed = time.time() - start_time
    print(f"0 {elapsed:.3f}")


if __name__ == "__main__":
    run_tests()