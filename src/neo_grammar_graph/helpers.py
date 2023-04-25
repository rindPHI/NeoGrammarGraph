import re
from typing import List

from orderedset import OrderedSet

from neo_grammar_graph.type_defs import Grammar

RE_NONTERMINAL = re.compile(r"(<[^<> ]*>)")


def split_expansion(expansion: str) -> List[str]:
    """
    Splits the given expansion alternative into tokens.

    >>> str(split_expansion("a<b><b>c<d>e"))
    "['a', '<b>', '<b>', 'c', '<d>', 'e']"

    :param expansion: The expansion alternative to split at nonterminal boundaries.
    :return: The separated terminal and nonterminal symbols in the expansion, in the
        original order.
    """

    return [token for token in re.split(RE_NONTERMINAL, expansion) if token]


def is_nonterminal(symbol: str) -> bool:
    """
    Checks whether the given symbol looks like a nonterminal symbol.

    >>> is_nonterminal("a")
    False

    >>> is_nonterminal("<a>")
    True

    >>> is_nonterminal("<a>a")
    False

    :param symbol: The grammar symbol to check.
    :return: True iff the given symbol is a nonterminal symbol.
    """

    return RE_NONTERMINAL.fullmatch(symbol) is not None


def nonterminals(expansion: str) -> OrderedSet[str]:
    """
    Extracts the set of nonterminals from the given expansion alternative.

    >>> str(nonterminals("a<b><b>c<d>e"))
    '{<b>, <d>}'

    :param expansion: A grammar expansion alternative.
    :return: The set of contained nonterminal symbols in the alternative.
    """

    return OrderedSet(RE_NONTERMINAL.findall(expansion))


def terminals(expansion: str) -> OrderedSet[str]:
    """
    Extracts the set of terminals from the given expansion alternative.

    >>> str(terminals("a<b><b>c<d>e"))
    '{a, c, e}'

    :param expansion: A grammar expansion alternative.
    :return: The set of contained terminal symbols in the alternative.
    """

    return OrderedSet(
        filter(lambda s: not is_nonterminal(s), split_expansion(expansion))
    )


def grammar_terminals(grammar: Grammar) -> OrderedSet[str]:
    """
    Returns all terminal symbols (tokens) from the given grammar.

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

    >>> str(grammar_terminals(grammar))
    '{ ; ,  := , a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}'

    :param grammar: The grammar to extract the terminal symbols from.
    :return: The terminal symbols from the grammar.
    """
    return OrderedSet(
        [
            terminal_symbol
            for key in grammar
            for expansion in grammar[key]
            for terminal_symbol in terminals(expansion)
        ]
    )
