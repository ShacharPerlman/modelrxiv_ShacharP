import time
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np

from Random_Graph_Models.Erdos_Renyi_Model import *
from Random_Graph_Models.Geometric_Model import *

pi = np.pi
exp = np.exp
real = np.real
imag = np.imag


class Graph:
    fig, ax = plt.subplots()

    def __init__(self, edgeMatrix: np.ndarray = None, edgeList: list[list[int]] = [], target_id: int = 0):
        """
        :param edgeMatrix: A square, binary matrix representing the graph.
        :type edgeMatrix: np.ndarray

        :param edgeList: An array of each node's neighbours in the graph, in order.
        :type edgeList: np.ndarray
        """
        self.edgeMatrix = edgeMatrix
        self.edgeList = edgeList
        self.target_id = target_id

        self.GenerateDualRepresentation()

    @property
    def V(self) -> int:
        """
        If the graph is represented by a matrix, then the number of nodes is the radical of the number of elements in
        the matrix representation of the graph. If the graph is given by an edge list then the number of nodes is just
        the length of the list.

        :return: The number of nodes in the graph.
        :rtype: int
        """
        if self.edgeMatrix is not None:
            return int(self.edgeMatrix.size ** 0.5)

        else:
            return len(self.edgeList)

    @property
    def E(self) -> int:
        """
        If the graph is initially represented as a matrix, then the number of edges is calculated as the number of 1, 's 
        in the edgeMatrix. These 1, 's represent a connection between node i and node j, and since the graph is undirected
        the matrix is symmetric. This means that the edges are exactly double counted because if there is an edge 
        between node i and node j, then in the matrix there will be a 1,  in both the (i,j)'th and the (j,i)'th entry. 
        To compensate for the double counting, the number of 1, 's is then divided by 2. 
        In the case that the graph is represented by an edgeList, the number of edges is the number of pairs (i,j) in 
        the list divided by 2. This counts the edges because these ordered pairs also double count the edges since any 
        edge between node i and node j will (in the list representation of the graph) require that the tuple (i,j) be 
        appended to the list of neighbours of node i, and the tuple (j,i) be appended to the list of neighbours of node
        j. Therefore, this will also double-count the edges in the graph and this count should also be divided by 2.
        
        :return: The number of edges in the graph. It is explicitly made into an int so pyCharm will be happy.  
        :rtype: int
        """
        if self.edgeMatrix is not None:
            return int(list(self.edgeMatrix.ravel()).count(1) / 2)

        else:
            return int(sum([len(n) for n in self.edgeList]) / 2)

    def GenerateDualRepresentation(self) -> None:
        if self.edgeMatrix is not None:
            for i in np.arange(self.V):
                neighbours = [j for j in np.arange(self.V) if self.edgeMatrix[i][j] == 1]
                self.edgeList.append(neighbours)

        else:
            self.edgeMatrix = np.zeros(shape=(self.V, self.V), dtype=int)

            for i in np.arange(self.V):
                for j in np.arange(self.V):
                    self.edgeMatrix[i][j] = 1 if j in self.edgeList[i] else 0

    def display(self) -> None:
        print(self.edgeMatrix)
        print()
        print(self.edgeList)
        print()

        # the graph can be displayed as connections between V'th roots of unity
        for k, neighbours in enumerate(self.edgeList):
            z = (exp(2 * pi * 1j * k / self.V) + 1 + 1j) / 2
            for i in neighbours:
                w = (exp(2 * pi * 1j * i / self.V) + 1 + 1j) / 2

                plt.plot(
                    [real(z), real(w)], [imag(z), imag(w)],
                    color="red" if k == self.target_id or i == self.target_id else "gray"
                )

            plt.scatter(x=real(z), y=imag(z), c="red" if k == self.target_id else "black")

        self.ax.set_aspect('equal')
        plt.show()

    # noinspection PyTypeChecker
    def LinearDisplay(self, deviation: float = 0.6) -> None:
        # Maybe use quadric or cubic Bézier curves as the edges between nodes in the representation.
        Path = mpath.Path
        self.edgeMatrix = np.zeros(shape=(self.V, self.V), dtype=int)

        for i in np.arange(self.V):
            for j in np.arange(self.V):
                anchor_pt1, control_pt, anchor_pt2 = (i, 0), ((i + j) / (2 * self.V), deviation), (j, 0)

                pp1 = mpatches.PathPatch(
                    path=Path([anchor_pt1, control_pt, anchor_pt2, anchor_pt1],
                              [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
                    fc="none",
                    transform=self.ax.transData
                )

                self.ax.add_patch(pp1)
                plt.show()


def main():
    times = [time.time(), time.time()]
    for i, d in enumerate([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]):
        current_time, previous_time = time.time(), times[i]
        current_diff, previous_diff = current_time - previous_time, times[i] - times[i-1]

        print(f"Currently the radius is {d = } at  time = "
              f"{round(current_time - previous_time, 2)}  at a  rate = "
              f"{round(current_diff - previous_diff, 2)}")
        times.append(round(current_diff))

        degrees = [np.sum(generate_geometric_graph(n=400, d=d, seed=1000+i)[0]) for i in np.arange(500)]
        plt.hist(degrees, histtype="step")

    times.append(round(current_diff, 2)); print(f"The times were {times} for a total of {np.sum(times)}")
    plt.show()

    # G = Graph(edgeMatrix=generate_geometric_graph(n=50, d=0.3, seed=111), target_id=0)
    # G.display()


if __name__ == "__main__":
    main()
