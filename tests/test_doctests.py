import doctest
import unittest

from neo_grammar_graph import gg


class TestDocstrings(unittest.TestCase):
    def test_gg(self):
        doctest_results = doctest.testmod(m=gg)
        self.assertFalse(doctest_results.failed)


if __name__ == "__main__":
    unittest.main()
