import string
import unittest

from neo_grammar_graph import NeoGrammarGraph
from neo_grammar_graph.nodes import SymbolicNode


class TestNeoGrammarGraph(unittest.TestCase):
    def test_convert_tree_for_grammar_with_epsilon_productions(self):
        # This test asserts that we can convert a derivation tree with an epsilon
        # expansion to a graph. This failed before since the graph contained choice
        # nodes without children if there was an epsilon expansion; now, we add
        # an empty terminal symbol as a child.

        scriptsize_c_grammar = {
            "<start>": ["<statement>"],
            "<statement>": [
                "<block>",
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
            ],
            "<id>": [c for c in string.ascii_lowercase],
        }

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


if __name__ == "__main__":
    unittest.main()
