import sys
from typing import Tuple

# At m[k][i] is are the paths (tuple of tuples) connecting i and k.
from graph_tool import Graph
from graph_tool.spectral import adjacency
from scipy.sparse import csr_matrix

PathMatrix = Tuple[Tuple[Tuple[Tuple[int, ...], ...], ...], ...]

# The entry adj[k][i] is nonzero iff. i is connected to k.
AdjacencyMatrix = Tuple[Tuple[int, ...], ...]

m = (
    (0, 0, 1),
    (1, 0, 0),
    (1, 1, 0),
)


def path_matrix_from_adjacency_matrix(adj_matrix: AdjacencyMatrix) -> PathMatrix:
    return tuple(
        tuple(
            ((col_idx, row_idx),) if entry else () for col_idx, entry in enumerate(row)
        )
        for row_idx, row in enumerate(adj_matrix)
    )


def path_from_adjacency_csr_matrix(adjacency_matrix: csr_matrix) -> PathMatrix:
    return tuple(
        tuple(
            ((col_idx, row_idx),)
            if col_idx in adjacency_matrix[row_idx].nonzero()[1]
            else ()
            for col_idx in range(adjacency_matrix.shape[1])
        )
        for row_idx in range(adjacency_matrix.shape[0])
    )


def pathmatmul(path_matrix_1: PathMatrix, path_matrix_2: PathMatrix) -> PathMatrix:
    def merge_paths(
        path_1: Tuple[int, ...], path_2: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        assert path_1[-1] == path_2[0], "Paths cannot be connected."
        return path_1[:-1] + path_2

    return tuple(
        tuple(
            tuple(
                merge_paths(path_2, path_1)
                for i in range(len(path_matrix_1))
                for path_1 in path_matrix_1[row_idx][i]
                for path_2 in path_matrix_2[i][col_idx]
            )
            for col_idx in range(len(path_matrix_2[0]))
        )
        for row_idx in range(len(path_matrix_1))
    )


graph = Graph()
v1, v2, v3, v4 = graph.add_vertex(4)

graph.add_edge(v1, v2)
graph.add_edge(v1, v3)
graph.add_edge(v2, v3)
graph.add_edge(v3, v4)
graph.add_edge(v4, v2)

pm = path_from_adjacency_csr_matrix(adjacency(graph))
print(pathmatmul(pm, pm))
sys.exit(0)

path_matrix = path_matrix_from_adjacency_matrix(m)
print(path_matrix)
print(pathmatmul(path_matrix, path_matrix))
print(pathmatmul(path_matrix, pathmatmul(path_matrix, path_matrix))[0][0])
print(
    pathmatmul(
        pathmatmul(path_matrix, path_matrix), pathmatmul(path_matrix, path_matrix)
    )
)
