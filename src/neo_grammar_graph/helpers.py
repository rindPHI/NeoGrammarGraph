import re
from typing import List

RE_NONTERMINAL = re.compile(r"(<[^<> ]*>)")


def split_expansion(expansion: str) -> List[str]:
    return [token for token in re.split(RE_NONTERMINAL, expansion) if token]
