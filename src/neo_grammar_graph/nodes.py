from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class Node(ABC):
    """
    Represents a node in the :class:`~neo_grammar_graph.GrammarGraph`. The
    :code:`ident` field allows distinguishing different nodes of the same
    type (e.g., the same nonterminal symbol in different contexts).

    Abstract base class.
    """

    ident: int


@dataclass(frozen=True)
class SymbolicNode(Node, ABC):
    """
    A symbolic node is a node corresponding to a grammar element, i.e., a nonterminal
    or terminal symbol in a specific context.
    """

    value: str

    def __str__(self):
        return f"{self.value} ({self.ident})"


@dataclass(frozen=True)
class NonterminalNode(SymbolicNode):
    """
    A node representing a nonterminal in a specific context.

    Example:

    >>> assgn_node_1 = NonterminalNode(0, "<assgn>")
    >>> assgn_node_2 = NonterminalNode(1, "<assgn>")

    >>> print(assgn_node_1)
    <assgn> (0)

    >>> print(assgn_node_2)
    <assgn> (1)

    >>> assgn_node_1 == NonterminalNode(0, "<assgn>")
    True

    >>> assgn_node_1 == assgn_node_2
    False

    >>> hash(assgn_node_1) == hash(NonterminalNode(0, "<assgn>"))
    True
    """

    pass


@dataclass(frozen=True)
class TerminalNode(SymbolicNode):
    """
    A node representing a terminal in a specific context; see also
    :class:`~neo_grammar_graph.nodes.NonterminalNode`.

    Example:

    >>> terminal_node = TerminalNode(0, "Hello World!")
    >>> print(terminal_node)
    'Hello World!' (0)
    """

    pass

    def __str__(self):
        return f"{repr(self.value)} ({self.ident})"


@dataclass(frozen=True)
class ChoiceNode(Node):
    """
    A "synthetic" choice node representing an alternation in the grammar.

    Example:

    >>> choice_node = ChoiceNode(0, "<assgn>")
    >>> print(choice_node)
    <assgn>-choice (0)
    """

    parent_nonterminal: str

    def __str__(self):
        return f"{self.parent_nonterminal}-choice ({self.ident})"
