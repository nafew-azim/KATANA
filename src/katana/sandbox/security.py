"""Static security screening for agent-generated candidate programs.

First line of defence of the Sandbox Execution Environment (Fig. 2):
candidates touching the OS, subprocesses, dynamic imports, serialisation,
or nested ``eval``/``exec`` are rejected outright with a 'security' failure
stage (reward -5.0) before any code is executed.
"""

import re

DANGEROUS_PATTERNS = [
    r"os\.",
    r"subprocess\.",
    r"__import__",
    r"eval\(",
    r"exec\s*\(",
    r"pickle\.",
    r"torch\.save\(",
]


def contains_dangerous_code(code: str) -> bool:
    return any(re.search(pattern, code) for pattern in DANGEROUS_PATTERNS)
