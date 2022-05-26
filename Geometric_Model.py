import numpy as np
import matplotlib.pyplot as plt


def GenerateGeometricGraph(n: int, d: float, seed: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    np.random.seed(seed)

    fig, ax = plt.subplots()

    edgeMatrix = np.zeros(shape=(n, n), dtype=int)
    samples_from_unit_square = [(np.random.random(), np.random.random()) for _ in np.arange(n)]

    for i, (px, py) in enumerate(samples_from_unit_square):
        for j, (qx, qy) in enumerate(samples_from_unit_square):
            if i < j:
                if pow(px - qx, 2) + pow(py - qy, 2) <= pow(d, 2):
                    edgeMatrix[i][j] = edgeMatrix[j][i] = 1
                    plt.plot([px, qx], [py, qy], color="red")
                # draw line between the i'th sample and the j'th sample

    X, Y = [u[0] for u in samples_from_unit_square], [u[1] for u in samples_from_unit_square]
    plt.scatter(X, Y)

    ax.set_aspect('equal')
    plt.show()

    return edgeMatrix, samples_from_unit_square


def generate_geometric_graph(n: int, d: float, seed: int) -> np.ndarray:
    np.random.seed(seed)

    nodes = np.random.rand(n, 2)
    graph = np.zeros([n, n])
    idx = np.triu_indices(n, k=1)
    graph.ravel()[np.ravel_multi_index(idx, (n, n))] = np.array(
        [d ** 2 > (nodes[i[0]][0] - nodes[i[1]][0]) ** 2 + (nodes[i[0]][1] - nodes[i[1]][1]) ** 2 for i in
         np.array(idx).T], dtype=int)
    return graph + graph.T
