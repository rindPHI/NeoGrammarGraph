from typing import Dict, Tuple

from graph_tool import Graph
from graph_tool.spectral import adjacency
from scipy.sparse import csr_matrix

Path = Tuple[int, ...]


class SparsePathMatrix:
    def __init__(
        self, entries: Dict[Tuple[int, int], Tuple[Path, ...]], shape: Tuple[int, int]
    ):
        self.entries = entries
        self.shape = shape

    @staticmethod
    def from_csr_adj_matrix(csr_adj_matrix: csr_matrix):
        entries = {}
        shape = (int(csr_adj_matrix.shape[1]), int(csr_adj_matrix.shape[0]))

        for y, x in zip(*csr_adj_matrix.nonzero()):
            x, y = int(x), int(y)
            entries[x, y] = ((x, y),) if csr_adj_matrix[y, x] else ()

        return SparsePathMatrix(entries, shape)

    def __matmul__(self, other: "SparsePathMatrix") -> "SparsePathMatrix":
        assert self.shape[0] == other.shape[1]
        k = self.shape[0]

        shape = (other.shape[0], self.shape[1])
        entries = {}

        def merge_paths(path_1: Path, path_2: Path) -> Path:
            assert path_1[-1] == path_2[0], "Paths cannot be connected."
            return path_1[:-1] + path_2

        def merge_path_sets(
            paths_1: Tuple[Path, ...], paths_2: Tuple[Path, ...]
        ) -> Tuple[Path, ...]:
            return tuple(
                merge_paths(path_1, path_2) for path_1 in paths_1 for path_2 in paths_2
            )

        for (x_1, y_1), paths_1 in self.entries.items():
            for (x_2, y_2), paths_2 in other.entries.items():
                if x_1 != y_2:
                    continue

                entries[x_2, y_1] = entries.get((x_2, y_1), ()) + merge_path_sets(
                    paths_2, paths_1
                )

        return SparsePathMatrix(entries, shape)

    def __getitem__(self, item):
        assert isinstance(item, tuple)
        return self.entries.get(item, [])

    def __str__(self):
        result = ""

        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                result += str(self.entries.get((col, row), ())) + " "
            result += "\n"

        return result[:-1]


graph = Graph()
v1, v2, v3, v4 = graph.add_vertex(4)

graph.add_edge(v1, v2)
graph.add_edge(v1, v3)
graph.add_edge(v2, v3)
graph.add_edge(v3, v4)
graph.add_edge(v4, v2)

matrix: csr_matrix = adjacency(graph)

spm = SparsePathMatrix.from_csr_adj_matrix(matrix)
print(spm.entries)
print(spm)
print()
twopower = spm @ spm
print(twopower @ twopower)
