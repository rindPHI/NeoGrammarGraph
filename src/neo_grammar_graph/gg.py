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
from functools import lru_cache
from typing import Dict, List, Callable, Optional, Tuple, Any

from graph_tool import Graph, Vertex, Edge, GraphView
from graph_tool.search import bfs_search, BFSVisitor, StopSearch
from graph_tool.topology import transitive_closure, all_paths
from orderedset import OrderedSet

from neo_grammar_graph.helpers import split_expansion, grammar_terminals, is_nonterminal
from neo_grammar_graph.type_defs import Grammar, ParseTree


class NeoGrammarGraph:
    def __init__(
        self,
        grammar: Grammar,
        _graph: Optional[Graph] = None,
        _closure: Optional[Graph] = None,
        _choice_nodes: Optional[OrderedSet[str]] = None,
        _nonterminal_vertex_map: Optional[Dict[str, Vertex]] = None,
    ):
        """
        Constructs a :class:`~grammar_graph.NeoGrammarGraph` from the given
        :class:`~grammar_graph.type_defs.Grammar`.

        :param grammar: The grammar to construct the
            :class:`~grammar_graph.NeoGrammarGraph` object from.
        :param _graph: An optional graph object. For internal purposes.
        :param _closure: An optional transitive closure of the passed graph object.
            For internal purposes.
        :param _choice_nodes: The choice nodes in the graph object. For internal
            purposes.
        :param _nonterminal_vertex_map: An optional mapping from grammar symbols to
            the corresponding vertex objects. For internal purposes.
        """

        # Either none or all of the optional arguments (but :code:`_closure`) should
        # be provided.
        assert (
            _graph is None
            and _closure is None
            and _choice_nodes is None
            and _nonterminal_vertex_map is None
            or _graph is not None
            and _choice_nodes is not None
            and _nonterminal_vertex_map is not None
        )

        self.grammar: Grammar = grammar
        self.graph: Graph = _graph or Graph()
        self.closure: Optional[Graph] = _closure
        self.choice_nodes: OrderedSet[str] = _choice_nodes or OrderedSet()
        self.nonterminal_vertex_map: Dict[str, Vertex] = _nonterminal_vertex_map or {}

        # Cache
        self.__hash: Optional[int] = None

        if _graph is None:
            self.__initialize_graph()

    def __initialize_graph(self):
        """
        Initializes the graph: Adds vertices and edges according to the grammar,
        registers choice nodes, sets the vertex labels to the corresponding grammar
        symbols, and populates the map from grammar symbols to vertices.

        :return: Nothing. Writes to :code:`self.choice_nodes`, :code:`self.graph`,
            and resets :code:`self.nonterminal_vertex_map`.
        """

        edges: List[Tuple[str, str, int]] = []
        for nonterminal in self.grammar:
            for nr, alternative in enumerate(self.grammar[nonterminal]):
                choice_node_name = f"{nonterminal}-choice-{nr + 1}"
                self.choice_nodes.add(choice_node_name)
                edges.append((nonterminal, choice_node_name, 1))
                edges.extend(
                    [
                        (choice_node_name, expansion_element, 1)
                        for expansion_element in split_expansion(alternative)
                    ]
                )

        edge_weights = self.graph.new_ep("double")
        self.graph.vp.label = self.graph.add_edge_list(
            edges, hashed=True, eprops=[edge_weights]
        )
        self.nonterminal_vertex_map = {
            self.graph.vp.label[v]: v for v in self.graph.vertices()
        }

    def __eq__(self, other: Any) -> bool:
        """
        Returns True if the other object is a NeoGrammarGraph object with the same
        grammar.

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

        A graph equals itself.

        >>> graph == graph
        True

        Two different graph objects are equal if they have the same grammar.

        >>> NeoGrammarGraph(grammar) == NeoGrammarGraph(grammar)
        True

        But it is different from a proper subgraph of itself.

        >>> NeoGrammarGraph(grammar) == NeoGrammarGraph(grammar).subgraph("<assgn>")
        False

        :param other: The object to compare.
        :return: True if the other object is a NeoGrammarGraph object with the same
            grammar.
        """

        return isinstance(other, NeoGrammarGraph) and self.grammar == other.grammar

    def __hash__(self):
        """
        Computes the hash of this grammar graph object based on the grammar. The hash
        is cached.

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

        The hashes of graph objects with the same grammar are equal.

        >>> hash(NeoGrammarGraph(grammar)) == hash(NeoGrammarGraph(grammar))
        True

        The hash of a sub graph is (in this example!) different.

        >>> hash(NeoGrammarGraph(grammar)) == \
                hash(NeoGrammarGraph(grammar).subgraph("<assgn>"))
        False

        The sub graph for :code:`<start>` is the same as the original graph, so the
        hashes are also identical.

        >>> hash(NeoGrammarGraph(grammar)) == \
                hash(NeoGrammarGraph(grammar).subgraph("<start>"))
        True

        :return: A has value for this graph.
        """

        if self.__hash is None:
            self.__hash = hash(
                tuple(
                    [
                        (nonterminal, tuple(expansions))
                        for nonterminal, expansions in self.grammar.items()
                    ]
                )
            )

        return self.__hash

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

    def nodes(self) -> List[str]:
        """
        Returns a list of all graph nodes, including the artificial choice nodes.

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

        Including terminal symbols and choice nodes, there are 86 nodes in the graph:

        >>> len(graph.nodes())
        86

        These are the first ten nodes:

        >>> graph.nodes()[:6]
        ['<start>', '<start>-choice-1', '<stmt>', '<stmt>-choice-1', '<assgn>', ' ; ']

        :return: All nodes in this grammar graph.
        """

        return [self.symbol(v) for v in self.graph.vertices()]

    def edges(self) -> List[Tuple[str, str]]:
        """
        Returns a list of all edges in this graph.

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

        >>> print("\\n".join(map(str, graph.edges()[:10])))
        ('<start>', '<start>-choice-1')
        ('<start>-choice-1', '<stmt>')
        ('<stmt>', '<stmt>-choice-1')
        ('<stmt>', '<stmt>-choice-2')
        ('<stmt>-choice-1', '<assgn>')
        ('<stmt>-choice-1', ' ; ')
        ('<stmt>-choice-1', '<stmt>')
        ('<assgn>', '<assgn>-choice-1')
        ('<stmt>-choice-2', '<assgn>')
        ('<assgn>-choice-1', '<var>')

        :return: All edges in the grammar graph. Contains edges to artificial
            "choice nodes."
        """

        return [
            (self.symbol(edge.source()), self.symbol(edge.target()))
            for edge in self.graph.edges()
        ]

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

    def subgraph(self, start_nonterminal: str) -> "NeoGrammarGraph":
        """
        Computes a sub graph for that part of the grammar starting with
        :code:`start_nonterminal`. The resulting grammar will still start with
        "<start>", but this initial nonterminal will be connected to
        :code:`start_nonterminal`.

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

        The nonterminals of the sub graph correspond to those reachable from the chosen
        start nonterminal. In particular, :code:`<stmt>` is missing in the below
        example:

        >>> subgraph = graph.subgraph("<assgn>")
        >>> list(subgraph.grammar.keys())
        ['<start>', '<assgn>', '<rhs>', '<var>', '<digit>']

        The :code:`<start>` nonterminal is connected to :code:`start_nonterminal`:

        >>> [(s, t) for s, t in subgraph.edges() if s == "<start>" or t == "<assgn>"]
        [('<start>', '<start>-choice-1'), ('<start>-choice-1', '<assgn>')]

        The only edge of a sub graph not contained in the original graph is part of
        the connection of :code:`<start>` to the :code:`start_nonterminal` (via the
        corresponding choice node):

        >>> set(subgraph.edges()).difference(set(graph.edges()))
        {('<start>-choice-1', '<assgn>')}

        Two arbitrary nodes in the subgraph are connected if, and only if, they are
        connected in the original graph:

        >>> for n_1 in subgraph.nodes():
        ...     for n_2 in subgraph.nodes():
        ...         assert (
        ...             not graph.reachable(n_1, n_2) and
        ...             not subgraph.reachable(n_1, n_2) or
        ...             graph.reachable(n_1, n_2) and
        ...             subgraph.reachable(n_1, n_2))

        The "sub graph" for the nonterminal symbol :code:`<start>` itself is the
        original graph:

        >>> graph is graph.subgraph("<start>")
        True

        :param start_nonterminal: The nonterminal symbol determining the sub grammar
            of the sub graph to be computed.
        :return: A sub graph for the grammar starting with :code:`start_nonterminal`.
            The resulting graph/grammar satisfy the convention that the grammar starts
            with the initial symbol :code:`<start>`, which is connected to
            :code:`start_nonterminal`.
        """

        # We always assume the start nonterminal of a grammar to be :code:`<start>`.
        # If this nonterminal is given, we return the original NeoGrammarGraph object.
        # Note that we do not copy the object, it will literally be the same one.
        if start_nonterminal == "<start>":
            return self

        start_vertex = self.vertex(start_nonterminal)
        assert start_vertex is not None

        # Create the new grammar using graph reachability
        new_grammar = {"<start>": [start_nonterminal]} | {
            nonterminal: list(expansion)
            for nonterminal, expansion in self.grammar.items()
            if nonterminal == start_nonterminal
            or self.reachable(start_nonterminal, nonterminal)
        }

        # Construct the reachability property used for filtering
        reachable = self.graph.new_vertex_property("bool")

        for vertex in self.graph.vertices():
            reachable[vertex] = vertex == start_vertex or self.reachable(
                start_nonterminal, self.graph.vp.label[vertex]
            )

        # Create the new graph-tool graph as a filtered version of the original one.
        new_graph = Graph(g=GraphView(self.graph, vfilt=reachable))
        # We make the view irreversible. Otherwise, we experience problems with some
        # functionality, such as DOT export.
        new_graph.purge_vertices()

        # We update the choice nodes set; all unreachable nodes are removed.
        new_choice_nodes = OrderedSet(
            [symbol for symbol in self.choice_nodes if reachable[self.vertex(symbol)]]
        )

        # We create a new "<start>" vertex with a single choice node pointing to
        # the chosen `start_nonterminal`. This is required since "<start>" needs to
        # be the start nonterminal by convention.
        new_start_vertex = new_graph.add_vertex()
        new_graph.vp.label[new_start_vertex] = "<start>"
        new_choice_vertex = new_graph.add_vertex()
        new_choice_label = "<start>-choice-1"
        new_graph.vp.label[new_choice_vertex] = new_choice_label
        new_choice_nodes.add(new_choice_label)

        # Since IDs might have changed, we update the nonterminal vertex map.
        new_nonterminal_vertex_map = {
            new_graph.vp.label[v]: v for v in new_graph.vertices()
        }

        # With the new vertex information, we add the edges connecting the new
        # initial nonterminal and the chosen start node.
        new_graph.add_edge(new_start_vertex, new_choice_vertex)
        new_graph.add_edge(
            new_choice_vertex, new_nonterminal_vertex_map[start_nonterminal]
        )

        return NeoGrammarGraph(
            new_grammar, new_graph, None, new_choice_nodes, new_nonterminal_vertex_map
        )

    def is_tree(self) -> bool:
        """
        Checks whether this graph is a tree, i.e., each node has exactly one parent.

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

        The assignment language grammar is recursive, so it cannot be a tree.

        >>> graph.is_tree()
        False

        Though assignments are not recursive, they are no trees: The :code:`<var>` node
        is reachable from an assignment or a :code:`<rhs>` nonterminal. The subgraph
        for :code:`<assgn>` is a directed, acyclic graph (DAG).

        >>> graph.subgraph("<assgn>").is_tree()
        False

        >>> graph.subgraph("<rhs>").is_tree()
        True

        :return:
        """

        class TreeVisitor(BFSVisitor):
            def __init__(self):
                self.result = True

            def non_tree_edge(self, e):
                self.result = False
                raise StopSearch()

        v = TreeVisitor()
        bfs_search(self.graph, source=self.vertex("<start>"), visitor=v)

        return v.result

    def filter(self, filter_function: Callable[[str], bool]) -> OrderedSet[str]:
        """
        Computes the set of graph nodes satisfying the criterion determined by
        :code:`filter_function`.

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

        Filtering out all choice nodes and all terminal symbols (nodes without children)
        yields all grammar nonterminals.

        >>> list(graph.filter(
        ...     lambda symbol:
        ...         symbol not in graph.choice_nodes
        ...         and graph.children(symbol)))
        ['<start>', '<stmt>', '<assgn>', '<var>', '<rhs>', '<digit>']

        On the other hand, we can also specifically ask for all symbols whose nodes
        don't have children, i.e., the terminal symbols:

        >>> list(graph.filter(lambda symbol: not graph.children(symbol)))[:14]
        [' ; ', ' := ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

        :param filter_function: A function taking a grammar symbol (nonterminals,
            terminals, or choice nodes) and deciding whether it should be filtered out
            (return False) or kept (return True).
        :return: All grammar symbols (potentially including choice nodes) satisfying
            the given filter criterion.
        """

        prop = self.graph.new_vertex_property("bool")

        for vertex in self.graph.vertices():
            prop[vertex] = filter_function(self.symbol(vertex))

        # Create the new graph-tool graph as a filtered version of the original one.
        return OrderedSet(
            map(self.symbol, Graph(g=GraphView(self.graph, vfilt=prop)).vertices())
        )

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

        # We sort paths by increasing length since we're interested in the shortest one.
        paths = sorted(paths, key=len)

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
        r"""
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

        >>> print("\n".join(map(
        ...     str, graph.paths_between("<stmt>", "<digit>", lambda idx, sym: True))))
        ('<stmt>', '<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>')
        ('<stmt>', '<stmt>-choice-2', '<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>')

        With the default node filter, those paths collapse to a single one:

        >>> str(graph.paths_between("<stmt>", "<digit>"))
        "{('<stmt>', '<assgn>', '<rhs>', '<digit>')}"

        There is no path from :code:`<digit>` to itself:

        >>> str(graph.paths_between("<digit>", "<digit>"))
        '{}'

        Path computation for self-reachable symbols works as expected.

        >>> str(graph.paths_between("<stmt>", "<stmt>"))
        "{('<stmt>', '<stmt>')}"

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

        return OrderedSet(
            [
                tuple(
                    [
                        self.symbol(self.graph.vertex(vid))
                        for idx, vid in enumerate(path)
                        if node_filter(idx, self.symbol(self.graph.vertex(vid)))
                    ]
                )
                for path in all_paths(self.graph, start_vertex, target_vertex)
            ]
        )

    @lru_cache(maxsize=None)
    def k_paths(
        self,
        k: int,
        up_to: bool = False,
        start_nonterminal: Optional[str] = None,
        include_terminals=True,
        graph: Optional[Graph] = None,
    ) -> OrderedSet[Tuple[str, ...]]:
        r"""
        This function computes all paths of length k in the graph, optionally restricted
        to those reachable from the nonterminal :code:`start_nonterminal`. If up_to
        is True, also smaller paths are considered. "Choice nodes" are not considered in
        the length computation, but are included in the returned results. k-paths ending
        in terminal symbols are included if, and only if, include_nonterminals is True.

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

        The 1-paths in the graph correspond to the nonterminal symbols (dictionary
        keys) in the underlying grammar.

        >>> str(graph.k_paths(1, include_terminals=False))
        "{('<start>',), ('<stmt>',), ('<assgn>',), ('<var>',), ('<rhs>',), ('<digit>',)}"

        We can ask for paths starting at a particular nonterminal.

        >>> print("\n".join(map(str, graph.k_paths(2, start_nonterminal="<digit>"))))
        ('<digit>', '<digit>-choice-1', '0')
        ('<digit>', '<digit>-choice-2', '1')
        ('<digit>', '<digit>-choice-3', '2')
        ('<digit>', '<digit>-choice-4', '3')
        ('<digit>', '<digit>-choice-5', '4')
        ('<digit>', '<digit>-choice-6', '5')
        ('<digit>', '<digit>-choice-7', '6')
        ('<digit>', '<digit>-choice-8', '7')
        ('<digit>', '<digit>-choice-9', '8')
        ('<digit>', '<digit>-choice-10', '9')

        Paths ending in terminal symbols can be excluded on demand.

        >>> print("\n".join(map(str, graph.k_paths(
        ...     3,
        ...     start_nonterminal="<assgn>",
        ...     include_terminals=False))))
        ('<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-1', '<var>')
        ('<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>')

        If up_to is set to True, we also obtain paths shorter than the set k.

        >>> print("\n".join(map(str, graph.k_paths(
        ...     3,
        ...     start_nonterminal="<assgn>",
        ...     up_to=True,
        ...     include_terminals=False))))
        ('<assgn>',)
        ('<assgn>', '<assgn>-choice-1', '<var>')
        ('<var>',)
        ('<rhs>', '<rhs>-choice-1', '<var>')
        ('<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-1', '<var>')
        ('<assgn>', '<assgn>-choice-1', '<rhs>')
        ('<rhs>',)
        ('<rhs>', '<rhs>-choice-2', '<digit>')
        ('<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>')
        ('<digit>',)

        For certain configurations, we might obtain an empty set of paths.

        >>> str(graph.k_paths(
        ...     4,
        ...     start_nonterminal="<assgn>",
        ...     include_terminals=False))
        '{}'

        :param k: The length of the paths to return. Maximal length if up_to is True,
            otherwise the exact lenght.
        :param up_to: Set to True iff you are interested also in paths shorter than k.
        :param start_nonterminal: If present, only k-paths in the part of the grammar
            reachable from this nonterminal will be considered.
        :param include_terminals: Set to True iff you are interested in paths ending
            in terminal symbols.
        :param graph: An optional Graph object if the paths in another graph (such as
            a ParseTree object) should be computed.
        :return: The set of paths according to the chosen parameters.
        """
        if graph is None:
            graph = self.graph

        # Each path of k terminal/nonterminal nodes includes k-1 choice nodes
        k += k - 1

        path_map: Dict[Vertex, List[Tuple[Vertex, ...]]] = {}

        class PathVisitor(BFSVisitor):
            def examine_edge(self, e: Edge):
                path_map.setdefault(e.source(), []).append((e.source(),))
                path_map.setdefault(e.target(), []).append((e.source(), e.target()))

                prefixes = [
                    path for path in path_map.get(e.source(), []) if len(path) < k
                ]
                if not prefixes:
                    return

                path_map[e.target()].extend(
                    [prefix + (e.target(),) for prefix in prefixes]
                )

        start_vertex = next(
            v
            for v in graph.vertices()
            if graph.vp.label[v]
            == (start_nonterminal if start_nonterminal is not None else "<start>")
        )

        bfs_search(graph, start_vertex, PathVisitor())

        all_terminals = OrderedSet()
        if not include_terminals:
            all_terminals = grammar_terminals(self.grammar)

        # We collect all the paths from `path_map` that do not start or end
        # in a choice node and have a suitable length. Furthermore, we eliminate
        # paths ending in terminal symbols if the corresponding flag is set.
        paths = [
            path
            for path_set in path_map.values()
            for path in path_set
            if graph.vp.label[path[0]] not in self.choice_nodes
            and graph.vp.label[path[-1]] not in self.choice_nodes
            and (up_to or len(path) == k)
            and graph.vp.label[path[-1]] not in all_terminals
        ]

        # For the final result, we convert vertex objects to grammar string symbols.
        return OrderedSet([tuple([graph.vp.label[v] for v in path]) for path in paths])

    def parse_tree_to_graph(self, tree: ParseTree) -> Graph:
        r"""
        Converts a ParseTree to a graph-tool Graph object.

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

        The following ParseTree can be derived from the above grammar:

        >>> tree: ParseTree = (
        ...   '<start>',
        ...     [('<stmt>',
        ...       [('<assgn>',
        ...         [('<var>', [('x', [])]),
        ...          (' := ', []),
        ...          ('<rhs>', [('<digit>', [('1', [])])])]),
        ...        (' ; ', []),
        ...        ('<stmt>',
        ...         [('<assgn>',
        ...           [('<var>', [('y', [])]),
        ...            (' := ', []),
        ...            ('<rhs>', [('<var>', [('x', [])])])])])])])

        We convert it to a graph...

        >>> tree_graph = graph.parse_tree_to_graph(tree)

        And assert that it is a tree.

        >>> class TreeVisitor(BFSVisitor):
        ...     def __init__(self):
        ...         self.result = True
        ...
        ...     def non_tree_edge(self, e):
        ...         self.result = False
        ...         raise StopSearch()
        >>> v = TreeVisitor()
        >>> bfs_search(tree_graph, tree_graph.vertex(0), v)
        >>> v.result
        True

        >>> print("\n".join(map(str, [
        ...     (tree_graph.vp.label[e.source()], tree_graph.vp.label[e.target()])
        ...     for e in tree_graph.edges()
        ... ])))
        ('<start>', '<start>-choice-1')
        ('<start>-choice-1', '<stmt>')
        ('<stmt>', '<stmt>-choice-1')
        ('<stmt>-choice-1', '<assgn>')
        ('<stmt>-choice-1', ' ; ')
        ('<stmt>-choice-1', '<stmt>')
        ('<assgn>', '<assgn>-choice-1')
        ('<stmt>', '<stmt>-choice-2')
        ('<assgn>-choice-1', '<var>')
        ('<assgn>-choice-1', ' := ')
        ('<assgn>-choice-1', '<rhs>')
        ('<var>', '<var>-choice-24')
        ('<rhs>', '<rhs>-choice-2')
        ('<stmt>-choice-2', '<assgn>')
        ('<assgn>', '<assgn>-choice-1')
        ('<var>-choice-24', 'x')
        ('<rhs>-choice-2', '<digit>')
        ('<digit>', '<digit>-choice-2')
        ('<assgn>-choice-1', '<var>')
        ('<assgn>-choice-1', ' := ')
        ('<assgn>-choice-1', '<rhs>')
        ('<var>', '<var>-choice-25')
        ('<rhs>', '<rhs>-choice-1')
        ('<digit>-choice-2', '1')
        ('<var>-choice-25', 'y')
        ('<rhs>-choice-1', '<var>')
        ('<var>', '<var>-choice-24')
        ('<var>-choice-24', 'x')

        :param tree: The parse tree to convert into a graph (with choice nodes).
        :return: A Graph object corresponding to a derivation in this grammar graph,
            including the right choice nodes.
        """

        graph = Graph()
        graph.vp.label = graph.new_vertex_property("string")

        root = graph.add_vertex()

        stack = [(tree, root)]
        while stack:
            (label, children), v = stack.pop(0)
            graph.vp.label[v] = label

            if not children:
                continue

            assert label not in self.choice_nodes
            choice_node = next(
                node
                for node in self.children(label)
                if self.children(node) == list(map(lambda child: child[0], children))
            )

            choice_node_vertex = graph.add_vertex()
            graph.vp.label[choice_node_vertex] = choice_node
            graph.add_edge(v, choice_node_vertex)

            for child in children:
                child_vertex = graph.add_vertex()
                stack.append((child, child_vertex))
                graph.add_edge(choice_node_vertex, child_vertex)

        return graph

    def k_paths_in_tree(
        self,
        tree: ParseTree,
        k: int,
        include_potential_paths=True,
        include_terminals=True,
    ) -> OrderedSet[Tuple[str, ...]]:
        r"""
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

        The following ParseTree can be derived from the above grammar:

        >>> tree: ParseTree = (
        ...   '<start>',
        ...     [('<stmt>',
        ...       [('<assgn>',
        ...         [('<var>', [('x', [])]),
        ...          (' := ', []),
        ...          ('<rhs>', [('<digit>', [('1', [])])])]),
        ...        (' ; ', []),
        ...        ('<stmt>',
        ...         [('<assgn>',
        ...           [('<var>', [('y', [])]),
        ...            (' := ', []),
        ...            ('<rhs>', [('<var>', [('x', [])])])])])])])

        >>> print("\n".join(map(str, graph.k_paths_in_tree(
        ...     tree, 3, include_potential_paths=False, include_terminals=True))))
        ('<start>', '<start>-choice-1', '<stmt>', '<stmt>-choice-1', '<assgn>')
        ('<start>', '<start>-choice-1', '<stmt>', '<stmt>-choice-1', ' ; ')
        ('<start>', '<start>-choice-1', '<stmt>', '<stmt>-choice-1', '<stmt>')
        ('<stmt>', '<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', '<var>')
        ('<stmt>', '<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', ' := ')
        ('<stmt>', '<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', '<rhs>')
        ('<stmt>', '<stmt>-choice-1', '<stmt>', '<stmt>-choice-2', '<assgn>')
        ('<assgn>', '<assgn>-choice-1', '<var>', '<var>-choice-24', 'x')
        ('<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>')
        ('<stmt>', '<stmt>-choice-2', '<assgn>', '<assgn>-choice-1', '<var>')
        ('<stmt>', '<stmt>-choice-2', '<assgn>', '<assgn>-choice-1', ' := ')
        ('<stmt>', '<stmt>-choice-2', '<assgn>', '<assgn>-choice-1', '<rhs>')
        ('<rhs>', '<rhs>-choice-2', '<digit>', '<digit>-choice-2', '1')
        ('<assgn>', '<assgn>-choice-1', '<var>', '<var>-choice-25', 'y')
        ('<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-1', '<var>')
        ('<rhs>', '<rhs>-choice-1', '<var>', '<var>-choice-24', 'x')

        >>> print("\n".join(map(str, graph.k_paths_in_tree(
        ...     tree, 3, include_potential_paths=False, include_terminals=False))))
        ('<start>', '<start>-choice-1', '<stmt>', '<stmt>-choice-1', '<assgn>')
        ('<start>', '<start>-choice-1', '<stmt>', '<stmt>-choice-1', '<stmt>')
        ('<stmt>', '<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', '<var>')
        ('<stmt>', '<stmt>-choice-1', '<assgn>', '<assgn>-choice-1', '<rhs>')
        ('<stmt>', '<stmt>-choice-1', '<stmt>', '<stmt>-choice-2', '<assgn>')
        ('<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-2', '<digit>')
        ('<stmt>', '<stmt>-choice-2', '<assgn>', '<assgn>-choice-1', '<var>')
        ('<stmt>', '<stmt>-choice-2', '<assgn>', '<assgn>-choice-1', '<rhs>')
        ('<assgn>', '<assgn>-choice-1', '<rhs>', '<rhs>-choice-1', '<var>')

        >>> tree: ParseTree = (
        ...   '<start>',
        ...     [('<stmt>',
        ...       [('<assgn>',
        ...         [('<var>', [('x', [])]),
        ...          (' := ', []),
        ...          ('<rhs>', [('<digit>', [('1', [])])])]),
        ...        (' ; ', []),
        ...        ('<stmt>',
        ...         [('<assgn>', None)])])])


        >>> print("\n".join(map(str, graph.k_paths_in_tree(
        ...     tree, 3, include_potential_paths=True, include_terminals=False))))

        :param tree:
        :param k:
        :param include_potential_paths:
        :param include_terminals:
        :return:
        """

        tree_graph = self.parse_tree_to_graph(tree)

        tree_k_paths = self.k_paths(
            k,
            graph=tree_graph,
            up_to=False,
            start_nonterminal="<start>",
            include_terminals=include_terminals,
        )

        if not include_potential_paths:
            return tree_k_paths

        result = OrderedSet(tree_k_paths)

        # Add to result:
        #
        # - All k-paths that start with a nonterminal reachable from a nonterminal
        #   of some open tree leaf.
        # - All grammar k-paths for which there is a non-empty suffix of a tree k-path
        #   ending in a nonterminal that is a prefix of the grammar k-path.

        leaf_nonterminals = [
            tree_graph.vp.label[v]
            for v in tree_graph.vertices()
            if is_nonterminal(tree_graph.vp.label[v])
        ]

        grammar_k_paths = self.k_paths(
            k,
            up_to=False,
            start_nonterminal="<start>",
            include_terminals=include_terminals,
        )

        for other_path in grammar_k_paths.difference(set(result)):
            if any(
                self.reachable(leaf_nonterminal, other_path[0])
                for leaf_nonterminal in leaf_nonterminals
            ):
                result.add(other_path)

        for other_path in grammar_k_paths.difference(set(result)):
            # Add other_path if there is a non-empty suffix of a tree k-path ending in
            # a nonterminal that is a prefix of other_path.
            # TODO
            pass

        assert False

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
