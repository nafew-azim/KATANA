"""Parsing of agent completions into (reasoning, code) pairs.

Candidates must wrap their rationale in ``<think>...</think>`` and their
executable strategy in a markdown ```` ```python ```` code block (the
official output contract of the search). Responses missing the code block
are rejected upstream with a flat penalty.
"""

import re
import textwrap
from typing import Optional, Tuple

_CODE_PATTERNS = (
    r"(?s)```python\s*(.*?)```",      # official markdown contract
    r"(?s)```\s*(.*?)```",            # bare fenced block fallback
    r"(?s)<code>(.*?)</code>",        # legacy tag fallback
)


def parse_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the reasoning and code sections from a completion."""
    code = None
    for pattern in _CODE_PATTERNS:
        code_match = re.search(pattern, response)
        if code_match:
            code = textwrap.dedent(code_match.group(1)).strip()
            break

    reasoning_match = re.search(r"(?s)<think>(.*?)</think>", response)
    reasoning = None
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        reasoning = re.sub(r"—", "-", reasoning)
        reasoning = re.sub(r"\t", " ", reasoning)
        reasoning = re.sub(r"(?<!\n)\n(?!\n)", " ", reasoning)

    return reasoning, code
