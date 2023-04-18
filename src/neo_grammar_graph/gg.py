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
from typing import Dict

from graph_tool import Graph, Vertex
from graph_tool.topology import transitive_closure

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

        start_vertex = self.nonterminal_vertex_map[to_nonterminal]
        target_vertex = self.closure.vertex(
            self.graph.vertex_index[self.nonterminal_vertex_map[from_nonterminal]]
        )

        return start_vertex in target_vertex.out_neighbors()

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
