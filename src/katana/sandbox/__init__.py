"""Isolated Sandbox Execution Environment for candidate programs."""

from .executor import SandboxExecutor  # noqa: F401
from .security import contains_dangerous_code  # noqa: F401
