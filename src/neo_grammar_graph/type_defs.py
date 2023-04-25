from abc import ABC, abstractmethod
from typing import List, Dict, Iterator, Optional

NonterminalType = str
Grammar = Dict[NonterminalType, List[str]]


class ParseTree(ABC):
    @classmethod
    def __subclasshook__(cls, C):
        return hasattr(C, "__iter__") and hasattr(C, "__getitem__")

    @abstractmethod
    def __iter__(self) -> Iterator[Optional[str | List["ParseTree"]]]:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item: int) -> Optional[str | List["ParseTree"]]:
        raise NotImplementedError
