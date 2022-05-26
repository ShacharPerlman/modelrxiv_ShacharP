import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np


class Graph:
    fig, ax = plt.subplots()

    def __init__(self, edgeMatrix: np.ndarray = None, edgeList: list[list[int]] = [], target_id: int = 0):
        self.edgeMatrix = edgeMatrix
        self.edgeList = edgeList
        self.target_id = target_id

        self.GenerateDualRepresentation()

    @property
    def V(self) -> int:
        if self.edgeMatrix is not None:
            return int(self.edgeMatrix.size ** 0.5)

        else:
            return len(self.edgeList)

    @property
    def E(self) -> int:
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
        for k, neighbours in enumerate(self.edgeList):
            z = (np.exp(2 * np.pi * 1j * k / self.V) + 1 + 1j) / 2
            for i in neighbours:
                w = (np.exp(2 * np.pi * 1j * i / self.V) + 1 + 1j) / 2

                plt.plot(
                    [np.real(z), np.real(w)], [np.imag(z), np.imag(w)],
                    color="red" if k == self.target_id or i == self.target_id else "gray"
                )

            plt.scatter(x=np.real(z), y=np.imag(z), c="red" if k == self.target_id else "black")

        self.ax.set_aspect('equal')
        plt.show()

    def LinearDisplay(self, deviation: float) -> None:
        Path = mpath.Path
        self.edgeMatrix = np.zeros(shape=(self.V, self.V), dtype=int)

        for i in np.arange(self.V):
            for j in np.arange(self.V):
                anchor_pt1, control_pt, anchor_pt2 = (i, 0), ((i + j) / (2 * self.V), deviation, (j, 0)

                pp1 = mpatches.PathPatch(
                    path=Path([anchor_pt1, control_pt, anchor_pt2, anchor_pt1],
                              [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
                    fc="none",
                    transform=self.ax.transData
                )

                self.ax.add_patch(pp1)
                plt.show()
