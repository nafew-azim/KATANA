"""KATANA: Knowledge-Aligned Topology-Aware Neural Agents.

An RL framework that autonomously discovers executable pruning algorithms
for vision-language models.

ECCV 2026. Nafew Azim, Mir Robab Warish Ali, Fuad Rahman, Nabeel Mohammed.
"""

__version__ = "1.0.0"

from .rewards.system import MultiObjectiveRewardSystem  # noqa: F401
from .sandbox.executor import SandboxExecutor  # noqa: F401
