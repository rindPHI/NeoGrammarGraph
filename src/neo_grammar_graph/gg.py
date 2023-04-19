# Copyright © 2023 CISPA Helmholtz Center for Information Security.
# Author: Dominic Steinhöfel.
#
# This file is part of NeoGrammarGraph.
#
# NeoGrammarGraph is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NeoGrammarGraph is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NeoGrammarGraph.  If not, see <http://www.gnu.org/licenses/>.

import os.path
from typing import Dict, List, Callable, Optional, Tuple

from graph_tool import Graph, Vertex, Edge
from graph_tool.search import bfs_search, BFSVisitor, StopSearch
from graph_tool.topology import transitive_closure, all_paths
from orderedset import OrderedSet

from neo_grammar_graph.helpers import split_expansion
from neo_grammar_graph.type_defs import Grammar


class NeoGrammarGraph:
    def __init__(self, grammar: Grammar):
        """
        Constructs a :class:`~grammar_graph.NeoGrammarGraph` from the given
        :class:`~grammar_graph.type_defs.Grammar`.

        :param grammar: The grammar to construct the
            :class:`~grammar_graph.NeoGrammarGraph` object from.
        """
        self.grammar = grammar
        self.graph = Graph()
        self.closure = None

        edges = []

        for nonterminal in self.grammar:
            for nr, alternative in enumerate(self.grammar[nonterminal]):
                choice_node_name = f"{nonterminal}-choice-{nr + 1}"
                edges.append((nonterminal, choice_node_name, 1))
                edges.extend(
                    [
                        (choice_node_name, expansion_element, 1)
                        for expansion_element in split_expansion(alternative)
                    ]
                )

        self.edge_weights = self.graph.new_ep("double")
        self.graph.vp.label = self.graph.add_edge_list(
            edges, hashed=True, eprops=[self.edge_weights]
        )
        self.nonterminal_vertex_map: Dict[str, Vertex] = {
            self.graph.vp.label[v]: v for v in self.graph.vertices()
        }

    def vertex_idx(self, symbol: str) -> Optional[int]:
        """
        Returns the vertex index for the given grammar symbol.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> graph.vertex_idx("<start>")
        0

        >>> graph.vertex_idx("<start>-choice-1")
        1

        >>> graph.vertex_idx("<stmt>")
        2

        >>> graph.vertex_idx("a")
        15

        >>> graph.vertex_idx("<function>") is None
        True

        :param symbol: The grammar symbol.
        :return: The index of the given grammar symbol in the graph, or None if the
            symbol does not exist in the graph.
        """

        vertex = self.vertex(symbol)
        if vertex is None:
            return None

        return self.graph.vertex_index[vertex]

    def vertex(self, symbol: str) -> Optional[Vertex]:
        """
        Returns the vertex object for the given grammar symbol.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> str(graph.vertex("<start>"))
        '0'

        >>> str(graph.vertex("<start>-choice-1"))
        '1'

        >>> str(graph.vertex("<stmt>"))
        '2'

        >>> str(graph.vertex("a"))
        '15'

        >>> graph.vertex("<function>") is None
        True

        :param symbol: The grammar symbol.
        :return: The vertex object for the given grammar symbol in the graph, or None
            if the symbol does not exist in the graph.
        """

        return self.nonterminal_vertex_map.get(symbol, None)

    def symbol(self, vertex: Vertex) -> str:
        """
        Returns the symbol (terminal or nonterminal symbols from the grammar or choice
        node identifiers) for the given graph vertex.

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> graph.symbol(graph.graph.vertex(0))
        '<start>'

        >>> graph.symbol(graph.graph.vertex(1))
        '<start>-choice-1'

        :param vertex: The vertex for which to return the associated symbol
        :return: The symbol associated to the given vertex.
        """

        return self.graph.vp.label[vertex]

    def children(self, symbol: str) -> Optional[List[str]]:
        """
        Returns the immediate children of a grammar symbol in the graph. If the given
        symbol does not exist in the graph, :code:`None` is returned.

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> graph.children("<stmt>")
        ['<stmt>-choice-1', '<stmt>-choice-2']

        >>> graph.children("<stmt>-choice-1")
        ['<assgn>', ' ; ', '<stmt>']

        >>> graph.children(" ; ")
        []

        >>> graph.children(" nope ") is None
        True

        :param symbol: The parent symbol.
        :return: The children of :code:`symbol` in the graph.
        """

        vertex = self.vertex(symbol)
        if vertex is None:
            return None

        return [self.symbol(child_vertex) for child_vertex in vertex.out_neighbors()]

    def reachable(self, from_nonterminal: str, to_nonterminal: str) -> bool:
        """
        Checks whether the nonterminal symbol :code:`to_nonterminal` is reachable
        in the graph from the nonterminal symbol :code:`from_nonterminal`. Note that
        reachability is not reflexive; a nonterminal is only reachable from itself if
        there is an actual path starting and ending at that nonterminal.

        On the first call to this method, the transitive closure of the grammar graph
        is computed once; for all further invocations, reachablility is computed based
        on the transitive closure, which works in constant time. Since grammar graphs
        are usually not very dense, computing the transitive closure on the first
        invocation is expected to perform bettern than conducting a search from scratch
        on the original graph for each invocation.

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> graph.reachable("<stmt>", "<stmt>")
        True

        >>> graph.reachable("<stmt>", "<assgn>")
        True

        >>> graph.reachable("<assgn>", "<stmt>")
        False

        >>> graph.reachable("<assgn>", "<assgn>")
        False

        :param from_nonterminal: The nonterminal starting from which reachability
            should be checked.
        :param to_nonterminal: The nonterminal that should be reached.
        :return: True iff :code:`to_nonterminal` is reachable from
            :code:`from_nonterminal`.
        """

        if self.closure is None:
            self.closure = transitive_closure(self.graph)

        start_vertex = self.vertex(to_nonterminal)
        target_vertex = self.closure.vertex(self.vertex_idx(from_nonterminal))

        return start_vertex in target_vertex.out_neighbors()

    def shortest_path(
        self,
        source_nonterminal: str,
        target_nonterminal: str,
        node_filter: Callable[[int, str], bool] = lambda idx, _: idx % 2 == 0,
    ) -> Optional[List[str]]:
        """
        Computest the shortest path between two nonterminals in the grammar graph
        using a breadth-first search. Note that a more general algorithm like
        Dijkstra's is not required here since all edge weights in grammar graphs
        are equal.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        The shortest path between :code:`<stmt>` and :code:`<digit>` in the above
        grammar consists of four nonterminals:

        >>> graph.shortest_path("<stmt>", "<digit>")
        ['<stmt>', '<assgn>', '<rhs>', '<digit>']

        If we relax the default :code:`node_filter`, we are also presented the
        "choice nodes:"

        >>> graph.shortest_path("<stmt>", "<digit>", lambda idx, node: True)
        ['<stmt>', '<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>']

        For unconnected nonterminals (grammar graphs are directed!), we obtain an empty
        list.

        >>> graph.shortest_path("<digit>", "<stmt>")
        []

        If a nonterminal is reachable from itself, the algorithm returns a list of
        length one. Call :meth:`shortest_non_trivial_path` for a path of length > 1
        starting and ending in :code:`source_nonterminal`.

        >>> graph.shortest_path("<stmt>", "<stmt>")
        ['<stmt>']

        Reachability is not reflexive: If there is no path of at least one edge between
        a nonterminal and itself, an empty list is returned.

        >>> graph.shortest_path("<digit>", "<digit>")
        []

        If either the source or the target symbol does not exist, :code:`None` is
        returned.
        >>> graph.shortest_path("<single-digit>", "<digit>") is None
        True

        Nodes in the result can be arbitrarily filtered; let's take out the source
        nonterminal:

        >>> graph.shortest_non_trivial_path(
        ...     "<stmt>", "<digit>", lambda idx, node: idx > 0)
        ['<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>']

        :param source_nonterminal: The start nonterminal for the computation of a
            shortest path.
        :param target_nonterminal: The destination nonterminal for the computation of
            a shortest path.
        :param node_filter: A function for filtering out elements of the returned path,
            if any. The function must accept two arguments, the first will be an
            integer, the position of the path element, and the second a string, the path
            element itself. Indices/positions start at 0 for the first element.
        :return: A possibly empty list of nonterminals connecting the source and target
            nonterminals in the graph. Any nonempty returned list begins with
            :code:`source_nonterminal` and ends with :code:`target_nonterminal`. If
            source and target nonterminals are identical and there is a connection from
            that nonterminal to itself, the returned list contains one element, which
            is that nonterminal.
        """

        source_vertex = self.vertex(source_nonterminal)
        target_vertex = self.vertex(target_nonterminal)

        if source_vertex is None or target_vertex is None:
            return None

        pred: Dict[Vertex, Vertex] = {}
        outer_self = self

        class ShortestPathVisitor(BFSVisitor):
            def examine_edge(self, e: Edge):
                if e.target() not in pred:
                    pred[e.target()] = e.source()
                    if outer_self.graph.vp.label[e.target()] == target_nonterminal:
                        raise StopSearch()

        bfs_search(
            self.graph,
            source_vertex,
            ShortestPathVisitor(),
        )

        if target_vertex not in pred:
            # Nonterminal is unreachable
            return []

        result: List[str] = [target_nonterminal]
        current_vertex = target_vertex
        while current_vertex != source_vertex:
            pred_vertex = pred[current_vertex]
            result.insert(0, self.graph.vp.label[pred_vertex])
            current_vertex = pred_vertex

        return [node for idx, node in enumerate(result) if node_filter(idx, node)]

    def shortest_non_trivial_path(
        self,
        source_nonterminal: str,
        target_symbol: str,
        node_filter: Callable[[int, str], bool] = lambda idx, _: idx % 2 == 0,
    ) -> Optional[List[str]]:
        """
        Returns the shortest path between the given symbols. If these are the same
        symbols and there *is* a path connecting the symbol and itself, a
        non-trivial path of at least length 2 is returned. We compute paths starting
        at the child choice nodes in that case; for this reason,
        :code:`source_nonterminal` *must* be a nonterminal symbol and not a terminal.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        The shortest path between :code:`<stmt>` and :code:`<digit>` in the above
        grammar consists of four nonterminals:

        >>> graph.shortest_non_trivial_path("<stmt>", "<digit>")
        ['<stmt>', '<assgn>', '<rhs>', '<digit>']

        If we relax the default :code:`node_filter`, we are also presented the
        "choice nodes:"

        >>> graph.shortest_non_trivial_path("<stmt>", "<digit>", lambda idx, node: True)
        ['<stmt>', '<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>']

        For unconnected nonterminals (grammar graphs are directed!), we obtain an empty
        list.

        >>> graph.shortest_non_trivial_path("<digit>", "<stmt>")
        []

        If a nonterminal is reachable from itself, the algorithm returns a list of
        at least two elements, other than :meth:`shortest_path`.

        >>> graph.shortest_non_trivial_path("<stmt>", "<stmt>")
        ['<stmt>', '<stmt>']

        Reachability is not reflexive: If there is no path of at least one edge between
        a nonterminal and itself, an empty list is returned.

        >>> graph.shortest_non_trivial_path("<digit>", "<digit>")
        []

        If either the source or the target symbol does not exist, :code:`None` is
        returned.
        >>> graph.shortest_non_trivial_path("<single-digit>", "<digit>") is None
        True

        Excluding the start node, e.g., can still lead to paths of length 1. This is
        not the standard behaviour, however, and no typical use case.

        >>> graph.shortest_non_trivial_path(
        ...     "<stmt>", "<stmt>", lambda idx, node: idx > 1)
        ['<stmt>']

        Filters can result in empty paths.

        >>> graph.shortest_non_trivial_path(
        ...     "<stmt>", "<stmt>", lambda idx, node: idx > 2)
        []

        We can also do so for longer paths.

        >>> graph.shortest_non_trivial_path("<stmt>", "<digit>", lambda idx, node: idx > 0)
        ['<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>']

        :param source_nonterminal: The source symbol for the computation of the
            shortest path.
        :param target_symbol: The target symbol to reach.
        :param node_filter: A function for filtering out elements of the returned path,
            if any. The function must accept two arguments, the first will be an
            integer, the position of the path element, and the second a string, the path
            element itself. Indices/positions start at 0 for the first element.
        :return: A possibly empty list of nonterminals connecting the source and target
            nonterminals in the graph. Any nonempty returned list begins with
            :code:`source_nonterminal` and ends with :code:`target_nonterminal` and
            consists of at least two elements.
        """

        source_vertex = self.vertex(source_nonterminal)
        target_vertex = self.vertex(target_symbol)

        if source_vertex is None or target_vertex is None:
            return None

        # Compute the shortest path starting from each child choice node.
        # Since we start at choice nodes with index 1 in the results,
        # we update the :code:`node_filter` function accordingly.
        paths: List[List[str]] = [
            self.shortest_path(
                self.symbol(child_vertex),
                target_symbol,
                lambda idx, symbol: node_filter(idx + 1, symbol),
            )
            for child_vertex in source_vertex.out_neighbors()
        ]

        # Filter out empty paths (from which the target is not reachable).
        paths = [path for path in paths if path]

        if not paths:
            return []

        sorted(paths, key=len)

        result: List[str]

        if not node_filter(0, source_nonterminal):
            # If the start node itself should not be included in the result, we can
            # take the path as-is (it was constructed with a choice node as starting
            # vertex).
            result = paths[0]
        else:
            # Otherwise, we prepend the source nonterminal.
            result = [source_nonterminal] + paths[0]

        assert (
            not node_filter(len(result) - 1, target_symbol)
            or result[-1] == target_symbol
        )

        return result

    def paths_between(
        self,
        start_symbol: str,
        target_symbol: str,
        node_filter: Callable[[int, str], bool] = lambda idx, _: idx % 2 == 0,
    ) -> Optional[OrderedSet[Tuple[str, ...]]]:
        """
        Returns a list of all paths between the given grammar symbols.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        There are two paths from :code:`<stmt>` to :code:`digit` if we omit the
        filtering of intermediate choice nodes:

        >>> str(graph.paths_between("<stmt>", "<digit>", lambda idx, sym: True))
        "{('<stmt>', '<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>'), ('<stmt>', '<stmt>-choice-2', '<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>')}"

        With the default node filter, those paths collapse to a single one:

        >>> str(graph.paths_between("<stmt>", "<digit>"))
        "{('<stmt>', '<assgn>', '<rhs>', '<digit>')}"

        There is no path from :code:`<digit>` to itself:

        >>> str(graph.paths_between("<digit>", "<digit>"))
        '{}'

        If a symbol does not exist, we obtain :code:`None`:

        >>> graph.paths_between("<some-digit>", "<digit>") is None
        True

        :param start_symbol: The start symbol of the desired paths.
        :param target_symbol: The target symbol of the desired paths.
        :param node_filter: A function for filtering out elements of the returned paths,
            if any. The function must accept two arguments, the first will be an
            integer, the position of the path element, and the second a string, the path
            element itself. Indices/positions start at 0 for the first element.
        :return: A list of all paths between the given grammar symbols or :code:`None`
            if start or target symbols do not exist in the grammar graph.
        """

        start_vertex = self.vertex(start_symbol)
        target_vertex = self.vertex(target_symbol)

        if start_vertex is None or target_vertex is None:
            return None

        return OrderedSet([
            tuple([
                self.symbol(self.graph.vertex(vid))
                for idx, vid in enumerate(path)
                if node_filter(idx, self.symbol(self.graph.vertex(vid)))
            ])
            for path in all_paths(self.graph, start_vertex, target_vertex)
        ])

    def save_to_dot(self, file_name: str) -> None:
        """
        Saves the graph as a DOT digraph that can be, e.g., exported to a PNG file
        using :code:`dot -Tpng dot_file_name.dot -o out.png`. If the given file name
        does not end in :code:`.dot`, this ending is appended to the file name.

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> graph.save_to_dot("/tmp/grammar-graph.dot")
        >>> import pathlib
        >>> contents = pathlib.Path("/tmp/grammar-graph.dot").read_text()
        >>> print(contents[:110])
        digraph G {
        0 [label="<start>"];
        1 [label="<start>-choice-1"];
        2 [label="<stmt>"];
        3 [label="<stmt>-choice-1"]

        >>> graph.save_to_dot("/tmp/no-dot-file.txt")
        >>> pathlib.Path("/tmp/no-dot-file.txt").exists()
        False

        >>> pathlib.Path("/tmp/no-dot-file.txt.dot").exists()
        True

        :param file_name: The path to the file into which the exported DOT code will
            be stored. A :code:`.dot` ending is appended if not already present.
        :return: Nothing. Stores a file as side effect.
        """

        assert not os.path.isdir(file_name)

        if not file_name.endswith(".dot"):
            file_name += ".dot"

        self.graph.save(file_name)
