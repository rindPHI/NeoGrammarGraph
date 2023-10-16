import itertools
import os
import pathlib
import tempfile
from functools import reduce
from itertools import zip_longest
from typing import cast, Tuple, List

import numpy
from bidict import MutableBidict, bidict
from graph_tool import Graph, Vertex, VertexPropertyMap, GraphView
from graph_tool.generation import graph_union
from graph_tool.search import DFSVisitor, dfs_search
from graph_tool.topology import all_paths
from orderedset import FrozenOrderedSet
from returns.functions import tap
from returns.maybe import Maybe, Some, Nothing
from returns.pipeline import is_successful
from returns.result import Success, Failure, Result

from neo_grammar_graph import NeoGrammarGraph, InvalidTreeException
from neo_grammar_graph.gg import leaves_in_graph, k_paths_in_graph
from neo_grammar_graph.helpers import is_nonterminal
from neo_grammar_graph.nodes import ChoiceNode, NonterminalNode, Node, SymbolicNode
from neo_grammar_graph.type_defs import ParseTree, Path


class DTree:
    __next_id: int = 0

    @staticmethod
    def next_id() -> int:
        DTree.__next_id += 1
        return DTree.__next_id - 1

    def __init__(
        self,
        grammar_graph: NeoGrammarGraph,
        tree_graph: Graph,
        vertex_to_node: MutableBidict[Vertex, Tuple[Node, int]],
        maybe_root: Maybe[Vertex] = Nothing,
    ):
        """
        TODO

        :param grammar_graph:
        :param tree_graph:
        :param vertex_to_node:
        :param maybe_root:
        """

        self.grammar_graph = grammar_graph
        self.tree_graph = tree_graph
        self.vertex_to_node = vertex_to_node
        self.root = maybe_root.lash(lambda _: Some(tree_graph.vertex(0))).unwrap()
        assert isinstance(self.vertex_to_node[self.root][0], SymbolicNode)

    @staticmethod
    def from_parse_tree(
        parse_tree: ParseTree, grammar_graph: NeoGrammarGraph
    ) -> Result["DTree", InvalidTreeException]:
        """
        This function converts a :class:`~neo_grammar_graph.type_defs.ParseTree` object
        to a graph-based derivation tree.

        Example
        -------

        Consider our running example "assignment language:"

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

        >>> parse_tree: ParseTree = (
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

        We convert it to a graph-based derivation tree.

        >>> dtree = DTree.from_parse_tree(parse_tree, graph).unwrap()

        The root node is always the vertex with identifier 0.

        >>> dtree.vertex_to_node[0]
        (NonterminalNode(ident=0, value='<start>'), 0)

        When trying to convert an invalid tree, we obtain a Failure:

        >>> from neo_grammar_graph.helpers import deep_str
        >>> deep_str(DTree.from_parse_tree(("<start>", [("<assgn>", None)]), graph))
        '<Failure: Could not find a choice node from <start> (0) to [<assgn>]>'

        :param parse_tree: The parse tree to convert.
        :param grammar_graph: The reference grammar graph.
        :return: A derivation tree with nodes conforming to the grammar graph.
        """

        assert parse_tree
        assert is_nonterminal(parse_tree[0])

        tree_graph = Graph()
        tree_graph.vp.label = tree_graph.new_vertex_property("string")
        tree_graph.ep.label = tree_graph.new_edge_property("int")
        vertex_to_node = bidict()

        root = tree_graph.add_vertex()

        def add_choice_node_and_children(choice_node: ChoiceNode) -> None:
            """
            This function adds the given choice node and all its children to the
            tree.

            :param choice_node: The choice node to add.
            :return: Nothing; the variables :code:`tree_graph` and
                :code:`vertex_to_node` are extended.
            """

            choice_node_vertex = tree_graph.add_vertex()
            identifier = DTree.next_id()
            tree_graph.vp.label[choice_node_vertex] = f"{identifier}: {choice_node}"
            vertex_to_node[choice_node_vertex] = (choice_node, identifier)

            tree_graph.add_edge(v, choice_node_vertex)

            for idx, (tree_child, graph_child) in enumerate(
                zip(children or [], grammar_graph.children(choice_node) or [])
            ):
                child_vertex = tree_graph.add_vertex()
                stack.append((tree_child, graph_child, child_vertex))
                edge = tree_graph.add_edge(choice_node_vertex, child_vertex)
                tree_graph.ep.label[edge] = idx

        stack = [(parse_tree, NonterminalNode(0, parse_tree[0]), root)]
        while stack:
            (_, children), node, v = stack.pop(0)

            identifier = DTree.next_id()
            tree_graph.vp.label[v] = f"{identifier}: {node}"
            vertex_to_node[v] = (node, identifier)

            if not children:
                continue

            maybe_choice_node: Maybe[ChoiceNode] = next(
                (
                    Some(cast(ChoiceNode, child))
                    for child in grammar_graph.children(node)
                    if reduce(
                        lambda bval, nodes: bval and nodes[0] == nodes[1],
                        zip_longest(
                            map(
                                lambda symbolic_node: symbolic_node.value,
                                grammar_graph.children(child),
                            ),
                            map(lambda grandchild: grandchild[0], children),
                            fillvalue=None,
                        ),
                        True,
                    )
                ),
                Nothing,
            )

            if not is_successful(maybe_choice_node):
                return Failure(
                    InvalidTreeException(
                        f"Could not find a choice node from {node} to ["
                        + ", ".join(map(lambda c: c[0], children))
                        + "]"
                    )
                )

            maybe_choice_node.map(tap(add_choice_node_and_children))

        return Success(DTree(grammar_graph, tree_graph, vertex_to_node))

    def value(self) -> str:
        """
        :return: The value (nonterminal or terminal symbol) of the tree's root node.
        """

        return cast(SymbolicNode, self.vertex_to_node[self.root][0]).value

    def node(self) -> Tuple[Node, int]:
        """
        TODO
        :return:
        """

        return self.vertex_to_node[self.root]

    def id(self) -> int:
        """
        :return: The value (nonterminal or terminal symbol) of the tree's root node.
        """

        return self.vertex_to_node[self.root][1]

    def children(self) -> "Tuple[DTree, ...]":
        """
        TODO
        :return:
        """

        return tuple(
            self.get_subtree(child)
            for child in Maybe.from_optional(
                next(iter(self.tree_graph.get_out_neighbors(self.root)), None)
            )
            .map(self.tree_graph.get_out_neighbors)
            .value_or([])
        )

    def __len__(self) -> int:
        return self.tree_graph.num_vertices()

    def get_subtree(self, path_or_new_root: Path | Vertex | numpy.int64) -> "DTree":
        """
        This method computes the subtree at the specified path or vertex. The underlying
        graph-tool tree is masked by a view; no nodes or edges are removed.

        Example
        -------

        We consider the assignment language grammar.

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

        The following ParseTree, representing the assignment sequence
        :code:`x := 1 ; y := x`, can be derived from that grammar:

        >>> parse_tree: ParseTree = (
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

        We convert it to a graph-based derivation tree.

        >>> dtree = DTree.from_parse_tree(parse_tree, graph).unwrap()

        At path (0, 2, 0), we find the second assignment:

        >>> dtree.vertex_to_node[dtree.vertex_at((0, 2, 0))][0].value
        '<assgn>'

        Let us obtain the subtree at that path:

        >>> subtree = dtree.get_subtree((0, 2, 0))
        >>> subtree.value()
        '<assgn>'

        The original derivation tree contains 29 vertices (including choice nodes):

        >>> len(dtree)
        29

        In the subtree starting at (0, 2, 0), we find 11 vertices:

        >>> len(subtree)
        11

        The new subtree has three children:

        >>> [c.value() for c in subtree.children()]
        ['<var>', ' := ', '<rhs>']

        We also can ask for the children of the first child:

        >>> [c.value() for c in subtree.get_subtree((0,)).children()]
        ['y']

        :param path_or_new_root: The new root vertex or the path to that vertex.
        :return: A subtree view for the tree starting at the specified root.
        """

        class VisitorExample(DFSVisitor):
            def __init__(self, discovered: VertexPropertyMap):
                self.discovered = discovered

            def discover_vertex(self, v: Vertex):
                self.discovered[v] = True

        discovered = self.tree_graph.new_vertex_property("bool")
        new_root = self.arg_to_vertex(path_or_new_root)

        dfs_search(self.tree_graph, new_root, VisitorExample(discovered))
        graph_view = GraphView(self.tree_graph, vfilt=discovered)

        return DTree(
            self.grammar_graph,
            graph_view,
            self.vertex_to_node,
            Some(new_root),
        )

    def vertex_at(self, path: Path) -> Vertex:
        """
        This method computes the vertex at the specified path.

        Example
        -------

        We consider the string :code:`x := 1 ; y := x` in our assignment language:

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

        >>> parse_tree: ParseTree = (
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

        >>> dtree = DTree.from_parse_tree(parse_tree, graph).unwrap()

        At path (0, 2, 0, 0, 0), we find the vertex corresponding to the identifier y:

        >>> dtree.vertex_to_node[dtree.vertex_at((0, 2, 0, 0, 0))][0]
        TerminalNode(ident=0, value='y')

        :param path: The path of the vertex to find.
        :return: The vertex at the specified path.
        """

        result: Vertex = self.root

        while path:
            idx, *path = path
            result = next(result.out_neighbors())  # Skip choice node
            result = next(itertools.islice(result.out_neighbors(), idx, None))

        return result

    def open_leaves(self) -> "Tuple[DTree, ...]":
        """
        This method returns the open leaves of this derivation tree, that is, the
        leaves marked with a nonterminal symbol.

        Example
        -------

        We consider the (incomplete/open) string :code:`x := <rhs> ; <var> := x` in
        our assignment language:

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

        >>> parse_tree: ParseTree = (
        ...   '<start>',
        ...     [('<stmt>',
        ...       [('<assgn>',
        ...         [('<var>', [('x', [])]),
        ...          (' := ', []),
        ...          ('<rhs>', None)]),
        ...        (' ; ', []),
        ...        ('<stmt>',
        ...         [('<assgn>',
        ...           [('<var>', None),
        ...            (' := ', []),
        ...            ('<rhs>', [('<var>', [('x', [])])])])])])])

        >>> dtree = DTree.from_parse_tree(parse_tree, graph).unwrap()

        We obtain the symbols of the two open subtrees as follows:

        >>> [subtree.value() for subtree in dtree.open_leaves()]
        ['<rhs>', '<var>']

        :return: The open subtrees of this tree.
        """

        return tuple(
            self.get_subtree(v)
            for v in leaves_in_graph(self.tree_graph)
            if isinstance(self.vertex_to_node[v][0], NonterminalNode)
        )

    def replace_subtree(
        self, path_or_vertex: Path | Vertex | numpy.int64, new_subtree: "DTree"
    ) -> "DTree":
        """
        Replaces the subtree at the given position with the new subtree.
        NOTE: No pruning/cleanup is performed if the specified node has children.
        They will be left dangling. Furthermore, we do not validate if the resulting
        tree is grammatically valid. This is up to the user.

        Example
        -------

        We consider the (incomplete/open) string :code:`x := <rhs> ; <var> := x` in
        our assignment language:

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

        >>> parse_tree: ParseTree = (
        ...   '<start>',
        ...     [('<stmt>',
        ...       [('<assgn>',
        ...         [('<var>', [('x', [])]),
        ...          (' := ', []),
        ...          ('<rhs>', None)]),
        ...        (' ; ', []),
        ...        ('<stmt>',
        ...         [('<assgn>',
        ...           [('<var>', None),
        ...            (' := ', []),
        ...            ('<rhs>', [('<var>', [('x', [])])])])])])])

        >>> dtree = DTree.from_parse_tree(parse_tree, graph).unwrap()

        In this tree, we want to replace the open :code:`<rhs>` leaf by a concrete
        :code:`<rhs> -> <digit> -> 3` instantiation:

        >>> new_subtree_parse_tree: ParseTree = ('<rhs>', [('<digit>', [("3", [])])])
        >>> new_subtree = DTree.from_parse_tree(new_subtree_parse_tree, graph).unwrap()

        >>> new_tree = dtree.replace_subtree((0, 0, 2), new_subtree)

        The length difference of the new and the original derivation tree equals the
        length of the subtree we inserted (minus one because we replace the original
        node at (0, 0, 2)):

        >>> len(new_subtree), len(new_tree) - len(dtree)
        (5, 4)

        The resulting tree looks as follows:

        >>> print(new_tree.to_str_repr())
        <start>
        └── <stmt>
            ├── <assgn>
            │   ├── <var>
            │   │   └── "x"
            │   ├── " := "
            │   └── <rhs>
            │       └── <digit>
            │           └── "3"
            ├── " ; "
            └── <stmt>
                └── <assgn>
                    ├── <var>
                    ├── " := "
                    └── <rhs>
                        └── <var>
                            └── "x"

        :param path_or_vertex: The path or vertex whose subtree (starting with itself)
            should be replaced.
        :param new_subtree: The new subtree.
        :return: The updated tree. The original trees are left untouched.
        """

        # I did not find a faster method to propagate the node identities...
        # Relying on the order of `vertex_to_node` did not work well. We might have
        # to optimize something here eventually. O(N) for tree replacement is not
        # acceptable.
        vtn_1 = self.tree_graph.new_vertex_property("object")
        for v1 in self.tree_graph.vertices():
            vtn_1[v1] = self.vertex_to_node[v1]

        vtn_2 = self.tree_graph.new_vertex_property("object")
        for v2 in new_subtree.tree_graph.vertices():
            vtn_2[v2] = new_subtree.vertex_to_node[v2]

        vertex = self.arg_to_vertex(path_or_vertex)
        intersection = new_subtree.tree_graph.new_vertex_property("int", val=-1)
        intersection[new_subtree.root] = int(vertex)

        union_graph: Graph
        new_props: List[VertexPropertyMap]
        union_graph, new_props = graph_union(
            self.tree_graph,
            new_subtree.tree_graph,
            intersection=intersection,
            props=[(vtn_1, vtn_2)]
            + list(
                zip(
                    self.tree_graph.properties.values(),
                    new_subtree.tree_graph.properties.values(),
                )
            ),
        )

        union_graph.vp.label, union_graph.ep.label = new_props[1:]

        vertex_to_node = bidict(
            {vertex: new_props[0][vertex] for vertex in union_graph.vertices()}
        )

        for v1, v2 in union_graph.edges():
            assert (
                isinstance(vertex_to_node[v1][0], SymbolicNode)
                and isinstance(vertex_to_node[v2][0], ChoiceNode)
                or isinstance(vertex_to_node[v2][0], SymbolicNode)
                and isinstance(vertex_to_node[v1][0], ChoiceNode)
            )

        return DTree(self.grammar_graph, union_graph, vertex_to_node)

    def k_paths(
        self,
        k: int,
        include_potential_paths=True,
        include_terminals=True,
    ) -> FrozenOrderedSet[Tuple[Node, ...]]:
        r"""
        This method computes the grammar k-paths covered by the given
        :class:`~neo_grammar_graph.type_defs.ParseTree`. The length of
        the paths is specified by the parameter "k." In the value for
        k, choice nodes are not included. That is, for k=3, paths of length
        5 (including two choice nodes) are computed. If :code:`include_potential_paths`
        is True, then k-paths reachable from open leaves in the tree are included.
        If :code:`include_terminals` is True, then paths ending in terminal symbols
        are included.

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

        >>> parse_tree: ParseTree = (
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
        >>> dtree = DTree.from_parse_tree(parse_tree, graph).unwrap()

        This is the set of all 3-paths in this tree.

        >>> from neo_grammar_graph.gg import paths_to_str
        >>> print(paths_to_str(dtree.k_paths(3, include_terminals=True)))
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <assgn> (0)
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), ' ; ' (0)
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <stmt> (1)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <var> (0)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), ' := ' (0)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0)
        <stmt> (0), <stmt>-choice (0), <stmt> (1), <stmt>-choice (1), <assgn> (1)
        <assgn> (0), <assgn>-choice (0), <var> (0), <var>-choice (23), 'x' (0)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <var> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), ' := ' (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <rhs> (0)
        <rhs> (0), <rhs>-choice (1), <digit> (0), <digit>-choice (1), '1' (0)
        <assgn> (1), <assgn>-choice (0), <var> (0), <var>-choice (24), 'y' (0)
        <assgn> (1), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)
        <rhs> (0), <rhs>-choice (0), <var> (1), <var>-choice (23), 'x' (0)

        Optionally, we can exclude the paths ending in terminal symbols:

        >>> print(paths_to_str(dtree.k_paths(3, include_terminals=False)))
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <assgn> (0)
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <stmt> (1)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <var> (0)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0)
        <stmt> (0), <stmt>-choice (0), <stmt> (1), <stmt>-choice (1), <assgn> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <var> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <rhs> (0)
        <assgn> (1), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)

        Let us "open" a tree leave by replacing a subtree with :code:`None`:

        >>> parse_tree: ParseTree = (
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
        ...           (' := ', []),
        ...           ('<rhs>', None)])])])])
        >>> dtree = DTree.from_parse_tree(parse_tree, graph).unwrap()

        First, let us inspect the "concrete" paths (without terminal symbols) in this
        cropped tree:

        >>> print(paths_to_str(dtree.k_paths(
        ...         3, include_potential_paths=False, include_terminals=False)))
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <assgn> (0)
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <stmt> (1)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <var> (0)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0)
        <stmt> (0), <stmt>-choice (0), <stmt> (1), <stmt>-choice (1), <assgn> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <var> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <rhs> (0)

        If we additionally ask for the potential paths, the two paths starting from the
        :code:`<assgn> (1)` node and passing the open :code:`<rhs> (0)` node are added
        to the result:

        >>> print(paths_to_str(dtree.k_paths(
        ...         3, include_potential_paths=True, include_terminals=False)))
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <assgn> (0)
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <stmt> (1)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <var> (0)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0)
        <stmt> (0), <stmt>-choice (0), <stmt> (1), <stmt>-choice (1), <assgn> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <var> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <rhs> (0)
        <assgn> (1), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)
        <assgn> (1), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        If we take the most "general" tree consisting only of an open root with the
        start nonterminal, the potential k-paths with terminal symbols are the same
        as all k-paths in the grammar graph.

        >>> DTree.from_parse_tree(("<start>", None), graph).unwrap().k_paths(3) \
        ...     == graph.k_paths(3)
        True

        >>> DTree.from_parse_tree(("<start>", None), graph).unwrap().k_paths(4) \
        ...     == graph.k_paths(4)
        True

        :param tree: The :class:`~neo_grammar_graph.type_defs.ParseTree` from which to
            extract the k-paths.
        :param k: The length of the paths.
        :param include_potential_paths: Set to True if you want to include potential
            paths that might be reachable for open tree leaves.
        :param include_terminals: Set to True if you want to include terminal symbols
            in the returned k-paths.
        :return: The grammar k-paths covered by the given tree.
        """

        tree_vertex_k_paths = k_paths_in_graph(
            self.tree_graph,
            lambda v: self.vertex_to_node[v][0],
            2 * k - 1,
            start_vertices=tuple(
                vertex
                for vertex in self.tree_graph.vertices()
                if isinstance(self.vertex_to_node[vertex][0], NonterminalNode)
            ),
            up_to=False,
            include_terminals=include_terminals,
            grammar_hash=hash(self.grammar_graph),
        )

        tree_k_paths = FrozenOrderedSet(
            [
                tuple(self.vertex_to_node[v][0] for v in path)
                for path in tree_vertex_k_paths
            ]
        )

        if not include_potential_paths:
            return tree_k_paths

        result = tree_k_paths

        # Add to result:
        #
        # - All k-paths that start with a nonterminal reachable from a nonterminal
        #   of some open tree leaf.
        # - All grammar k-paths for which there is a non-empty suffix of a tree k-path
        #   ending in a nonterminal that is a prefix of the grammar k-path.

        tree_leaves = leaves_in_graph(self.tree_graph)
        open_leaf_vertices = [
            v
            for v in tree_leaves
            if isinstance(self.vertex_to_node[v][0], NonterminalNode)
        ]

        open_leaf_nodes = [self.vertex_to_node[v][0] for v in open_leaf_vertices]

        grammar_k_paths = self.grammar_graph.k_paths(
            k,
            up_to=False,
            include_terminals=include_terminals,
        )

        for other_path in grammar_k_paths.difference(result):
            if any(
                other_path[0] == leaf_node
                or self.grammar_graph.reachable(leaf_node, other_path[0])
                for leaf_node in open_leaf_nodes
            ):
                result = FrozenOrderedSet(result | {other_path})

        tree_paths_to_open_leaves = FrozenOrderedSet(
            [
                tuple([self.vertex_to_node[v][0] for v in path])
                for open_leaf_vertex in open_leaf_vertices
                for path in all_paths(
                    self.tree_graph, self.tree_graph.vertex(0), open_leaf_vertex
                )
            ]
        )

        # Add other_path if there is a non-empty suffix of a tree path ending in
        # a nonterminal that is a prefix of other_path.
        #
        # Example:
        #
        # - other_path (length 2*k-1):            1234567
        # - tree_path (length arbitrary): 9999999912345
        # - Shared pre/postfix: 12345 (for shift of 2)
        #
        # `shift` can proceed in steps of 2, since we do not consider paths starting
        # or ending in choice nodes (every second node is a choice node).
        for other_path in grammar_k_paths.difference(result):
            for tree_path in tree_paths_to_open_leaves:
                for shift in range(1, 2 * k - 1, 2):
                    if tree_path[-shift:] == other_path[:shift]:
                        result = FrozenOrderedSet(result | {other_path})

        return result

    def save_to_dot(self, file_name: str) -> None:
        """
        Saves the tree as a DOT digraph that can be, e.g., exported to a PNG file
        using :code:`dot -Tpng dot_file_name.dot -o out.png`. If the given file name
        does not end in :code:`.dot`, this ending is appended to the file name.

        Example
        -------

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

        >>> parse_tree: ParseTree = (
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

        >>> graph = NeoGrammarGraph(grammar)
        >>> DTree._DTree__next_id = 0
        >>> dtree = DTree.from_parse_tree(parse_tree, graph).unwrap()

        >>> dtree.save_to_dot("/tmp/tree-graph.dot")
        >>> import pathlib
        >>> contents = pathlib.Path("/tmp/tree-graph.dot").read_text()
        >>> print(contents[:135])
        digraph G {
        0 [label="0: <start> (0)"];
        1 [label="1: <start>-choice (0)"];
        2 [label="2: <stmt> (0)"];
        3 [label="3: <stmt>-choice (0)"];

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

        self.tree_graph.save(file_name)

    def to_dot(self) -> str:
        """
        TODO

        :return:
        """

        with tempfile.NamedTemporaryFile(suffix=".dot") as tmp:
            self.save_to_dot(tmp.name)
            return pathlib.Path(tmp.name).read_text()

    def arg_to_vertex(self, path_or_new_root: Vertex | numpy.int64 | Path) -> Vertex:
        """
        Converts the given argument to a graph-tool vertex.

        :param path_or_new_root: Either a vertex (as an object or numpy int) or a path
            to a vertex.
        :return: The corresponding :class:`~graph_tool.Vertex` object.
        """

        return (
            path_or_new_root
            if isinstance(path_or_new_root, Vertex)
            else (
                self.tree_graph.vertex(path_or_new_root)
                if isinstance(path_or_new_root, numpy.int64)
                or isinstance(path_or_new_root, int)
                else self.vertex_at(path_or_new_root)
            )
        )

    def to_str_repr(self, level: int = 0) -> str:
        """
        This method converts a derivation tree to a textual representation capturing
        the whole structure.

        Example
        -------

        We consider the string :code:`x := 1 ; y := x` in our assignment language:

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

        >>> parse_tree: ParseTree = (
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

        The string representation looks as follows:

        >>> print(DTree.from_parse_tree(parse_tree, graph).unwrap().to_str_repr())
        <start>
        └── <stmt>
            ├── <assgn>
            │   ├── <var>
            │   │   └── "x"
            │   ├── " := "
            │   └── <rhs>
            │       └── <digit>
            │           └── "1"
            ├── " ; "
            └── <stmt>
                └── <assgn>
                    ├── <var>
                    │   └── "y"
                    ├── " := "
                    └── <rhs>
                        └── <var>
                            └── "x"

        :return:
        """

        result = (
            self.value()
            if isinstance(self.node()[0], NonterminalNode)
            else f'"{self.value()}"'
        )

        for child_idx, child in enumerate(self.children()):
            child_lines = child.to_str_repr(level + 1).split("\n")
            for line_idx, line in enumerate(child_lines):
                if not line_idx and child_idx < len(self.children()) - 1:
                    result += "\n" + "\u251c\u2500\u2500 " + line
                elif not line_idx and child_idx == len(self.children()) - 1:
                    result += "\n" + "\u2514\u2500\u2500 " + line
                elif child_idx == len(self.children()) - 1:
                    result += "\n" + "    " + line
                else:
                    result += "\n" + "\u2502   " + line

        return result
