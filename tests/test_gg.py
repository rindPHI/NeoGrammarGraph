import string
import unittest

from neo_grammar_graph import NeoGrammarGraph
from neo_grammar_graph.gg import paths_to_str
from neo_grammar_graph.nodes import SymbolicNode, NonterminalNode

scriptsize_c_grammar = {
    "<start>": ["<statement>"],
    "<statement>": [
        "<block>",
        "if<paren_expr> <statement> else <statement>",
        "if<paren_expr> <statement>",
        "while<paren_expr> <statement>",
        "do <statement> while<paren_expr>;",
        "<expr>;",
        ";",
    ],
    "<block>": ["{<statements>}"],
    "<statements>": ["<block_statement><statements>", ""],
    "<block_statement>": ["<statement>", "<declaration>"],
    "<declaration>": ["int <id> = <expr>;", "int <id>;"],
    "<paren_expr>": ["(<expr>)"],
    "<expr>": [
        "<id> = <expr>",
        "<test>",
    ],
    "<test>": [
        "<sum> < <sum>",
        "<sum>",
    ],
    "<sum>": [
        "<sum> + <term>",
        "<sum> - <term>",
        "<term>",
    ],
    "<term>": [
        "<paren_expr>",
        "<id>",
        "<int>",
    ],
    "<id>": [c for c in string.ascii_lowercase],
    "<int>": [
        "<digit_nonzero><digits>",
        "<digit>",
    ],
    "<digits>": [
        "<digit><int>",
        "<digit>",
    ],
    "<digit>": [d for d in string.digits],
    "<digit_nonzero>": [d for d in string.digits if d != "0"],
}


class TestNeoGrammarGraph(unittest.TestCase):
    def test_convert_tree_for_grammar_with_epsilon_productions(self):
        # This test asserts that we can convert a derivation tree with an epsilon
        # expansion to a graph. This failed before since the graph contained choice
        # nodes without children if there was an epsilon expansion; now, we add
        # an empty terminal symbol as a child.

        graph = NeoGrammarGraph(scriptsize_c_grammar)

        # All leaves should be symbolic nodes, i.e., no choice nodes.
        leaves = graph.filter(lambda node: not graph.children(node))
        self.assertTrue(all(isinstance(leaf, SymbolicNode) for leaf in leaves))

        tree = (
            "<start>",
            [
                (
                    "<statement>",
                    [
                        (
                            "<block>",
                            [
                                ("{", []),
                                (
                                    "<statements>",
                                    [
                                        (
                                            "<block_statement>",
                                            [
                                                (
                                                    "<declaration>",
                                                    [
                                                        ("int ", []),
                                                        ("<id>", [("x", [])]),
                                                        (";", []),
                                                    ],
                                                )
                                            ],
                                        ),
                                        ("<statements>", [("", [])]),
                                    ],
                                ),
                                ("}", []),
                            ],
                        )
                    ],
                )
            ],
        )

        graph.parse_tree_to_graph(tree)  # No exception

    def test_k_paths_all_ref_nodes(self):
        grammar = {
            "<start>": ["<stmt>"],
            "<stmt>": ["<assgn> ; <stmt>", "<assgn>"],
            "<assgn>": ["<var> := <rhs>"],
            "<rhs>": ["<var>", "<digit>"],
            "<var>": list(string.ascii_lowercase),
            "<digit>": list(string.digits),
        }
        graph = NeoGrammarGraph(grammar)

        paths = graph.k_paths(
            3, include_terminals=False, start_nodes=("<stmt>", "<assgn>")
        )

        stmt_0_paths = [
            path for path in paths if path[0] == NonterminalNode(0, "<stmt>")
        ]
        stmt_1_paths = [
            path for path in paths if path[0] == NonterminalNode(1, "<stmt>")
        ]
        self.assertEqual(
            [path[1:] for path in stmt_0_paths], [path[1:] for path in stmt_1_paths]
        )

        assgn_0_paths = [
            path for path in paths if path[0] == NonterminalNode(0, "<assgn>")
        ]
        assgn_1_paths = [
            path for path in paths if path[0] == NonterminalNode(1, "<assgn>")
        ]
        self.assertEqual(
            [path[1:] for path in assgn_0_paths], [path[1:] for path in assgn_1_paths]
        )


if __name__ == "__main__":
    unittest.main()
