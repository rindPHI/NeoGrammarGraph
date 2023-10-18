import re
from typing import List, Any, Callable, Sequence, TypeVar

import returns
from frozendict import frozendict
from orderedset import OrderedSet
from returns.maybe import Maybe, Some
from returns.result import Result, Success, Failure

from neo_grammar_graph.type_defs import Grammar, CanonicalGrammar

T = TypeVar("T")

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

    return [token for token in RE_NONTERMINAL.split(expansion) if token]


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


def canonical(grammar: Grammar) -> CanonicalGrammar:
    """
    This function converts a grammar to a "canonical" form in which terminals and
    nonterminals in expansion alternatives are split.

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

    Before conversion, there are two entries for :code:`<stmt>` including sequences
    of (non-)terminals:

    >>> print(grammar["<stmt>"])
    ['<assgn> ; <stmt>', '<assgn>']

    After conversion, the entries are lists of individual (non)-terminals:

    >>> print(canonical(grammar)["<stmt>"])
    [['<assgn>', ' ; ', '<stmt>'], ['<assgn>']]

    :param grammar: The grammar to convert.
    :return: The converted canonical grammar.
    """
    return {
        k: [split_expansion(expression) for expression in alternatives]
        for k, alternatives in grammar.items()
    }


def star(f: Callable[[[Any, ...]], T]) -> Callable[[Sequence[Any]], T]:
    return lambda x: f(*x)


def deep_str(obj: Any) -> str:
    """
    This function computes a "deep" string representation of :code:`obj`. This means
    that it also (recursively) invokes :code:`__str__` on all the elements of a list,
    tuple, set, OrderedSet, dict, or Success/Failure container (from the returns
    library).

    Example:
    --------

    We constuct a simple class with different :code:`__str__` and :code:`__repr__`
    implementations:

    >>> class X:
    ...     def __str__(self):
    ...         return "'An X'"
    ...     def __repr__(self):
    ...         return "X()"

    Invoking :code:`str` returns a "shallow" string representation:

    >>> str((X(), X()))
    '(X(), X())'

    Invoking :code:`deep_str` also converts the elements of the tuple to a string:

    >>> deep_str((X(), X()))
    "('An X', 'An X')"

    This also works for nested collections, such as a tuple in a list:

    >>> deep_str([(X(),)])
    "[('An X',)]"

    It also works for dictionaries...

    >>> deep_str({X(): [X()]})
    "{'An X': ['An X']}"

    ...frozen dictionaries...

    >>> deep_str(frozendict({X(): [X()]}))
    "{'An X': ['An X']}"

    ...sets...

    >>> deep_str({(X(),)})
    "{('An X',)}"

    ...frozen sets...

    >>> deep_str(frozenset({(X(),)}))
    "{('An X',)}"

    ...and ordered sets.

    >>> deep_str(OrderedSet({(X(),)}))
    "{('An X',)}"

    As a special gimick, the function also works for the returns library's Success
    and Failure containers:

    >>> deep_str(returns.result.Success([X(), X()]))
    "<Success: ['An X', 'An X']>"

    >>> deep_str(returns.result.Failure([X(), X()]))
    "<Failure: ['An X', 'An X']>"

    If the string representation of an object is empty, its :code:`repr` is returned:

    >>> str(StopIteration())
    ''

    >>> deep_str(StopIteration())
    'StopIteration()'

    :param obj: The object to recursively convert into a string.
    :return: A "deep" string representation of :code:`obj`.
    """

    if isinstance(obj, tuple):
        return (
            "(" + ", ".join(map(deep_str, obj)) + ("," if len(obj) == 1 else "") + ")"
        )
    elif isinstance(obj, list):
        return "[" + ", ".join(map(deep_str, obj)) + "]"
    elif (
        isinstance(obj, set)
        or isinstance(obj, OrderedSet)
        or isinstance(obj, frozenset)
    ):
        return "{" + ", ".join(map(deep_str, obj)) + "}"
    elif isinstance(obj, dict) or isinstance(obj, frozendict):
        return (
            "{"
            + ", ".join([f"{deep_str(a)}: {deep_str(b)}" for a, b in obj.items()])
            + "}"
        )
    elif isinstance(obj, Maybe):
        match obj:
            case Some(elem):
                return str(Some(deep_str(elem)))
            case returns.maybe.Nothing:
                return str(obj)
    elif isinstance(obj, Result):
        match obj:
            case Success(inner):
                return str(Success(deep_str(inner)))
            case returns.result.Failure(inner):
                return str(Failure(deep_str(inner)))
    elif not str(obj):
        return repr(obj)
    else:
        return str(obj)
