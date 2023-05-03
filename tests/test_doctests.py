import doctest
import unittest

from neo_grammar_graph import gg, helpers, nodes


class TestDocstrings(unittest.TestCase):
    def test_gg(self):
        doctest_results = doctest.testmod(m=gg)
        self.assertFalse(doctest_results.failed)

    def test_helpers(self):
        doctest_results = doctest.testmod(m=helpers)
        self.assertFalse(doctest_results.failed)

    def test_nodes(self):
        doctest_results = doctest.testmod(m=nodes)
        self.assertFalse(doctest_results.failed)


if __name__ == "__main__":
    unittest.main()
