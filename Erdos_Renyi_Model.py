import numpy as np


# rework so that there is no direct iteration (remove `for` loops)
def GenerateERGraph(n: int, p: float, seed: int) -> np.ndarray:
    np.random.seed(seed)

    edgeMatrix = np.zeros(shape=(n, n), dtype=int)

    for i in np.arange(n):
        for j in np.arange(n):
            if i < j:
                if np.random.choice(a=[0, 1], p=[1 - p, p]):
                    edgeMatrix[i][j] = edgeMatrix[j][i] = 1

    return edgeMatrix


def generate_er_graph(n: int, p: float, seed: int) -> np.ndarray:
    np.random.seed(seed)

    graph = np.zeros([n, n])
    idx = np.triu_indices(n, k=1)
    graph.ravel()[np.ravel_multi_index(idx, (n, n))] = np.array(np.random.rand(len(idx[0])) < p, dtype=int)
    return graph + graph.T
