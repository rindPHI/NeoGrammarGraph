from abc import abstractmethod
from typing import List, Dict, Iterator, Optional, Protocol, Sequence

NonterminalType = str
Grammar = Dict[NonterminalType, List[str]]


class ParseTree(Protocol):
    """
    A parse tree is a nested structure containing node labels and optional children.
    For example:

    >>> tree: ParseTree = ("<start>", [("<A>", None), ("x", [])])

    is a parse tree.

    In this tree, :code:`("<A>", None)` is an *open* leaf since the label is a
    nonterminal symbol and the children are :code:`None`, and :code:`("x", [])`
    if a *closed* leaf since the label is a terminal symbol and the children are
    an empty list.

    Only these combinations for leaves are considered legal. For example, a terminal
    symbol with a children element that is not an empty sequence is not a valid
    parse tree.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Optional[str | Sequence["ParseTree"]]]:
        ...

    @abstractmethod
    def __getitem__(self, item: int) -> Optional[str | Sequence["ParseTree"]]:
        ...
