"""Adaptive curriculum timeline (Fig. 2, top).

The search proceeds through four phases — Base Exploration, Balancing,
Optimization, Aggressive Optimization — that systematically shift the
agent's focus. Each phase carries its own (alpha_phi, beta_phi, tau_phi)
performance weighting (Eq. 3): early phases prioritise latency reduction
(alpha_0 = 0.1, beta_0 = 0.9) with a relaxed accuracy threshold to permit
aggressive topological exploration, while late phases strictly enforce
multimodal quality preservation (alpha_3 = 0.9, beta_3 = 0.1).

Advancement is performance-gated: once the recent compression performance
clears the current phase threshold, the phase index increments and the
parameter-retention target tightens by 20% (floored at ``min_param_target``).
"""

from typing import Deque, Dict

import numpy as np

# (alpha_phi, beta_phi, tau_phi) per curriculum phase — Eq. 3.
PHASE_PERFORMANCE_CONFIG = [
    (0.1, 0.9, 0.8),   # Phase 1: Base Exploration
    (0.3, 0.7, 0.9),   # Phase 2: Balancing
    (0.7, 0.3, 1.1),   # Phase 3: Optimization
    (0.9, 0.1, 1.3),   # Phase 4: Aggressive Optimization
]

# Per-phase multiplicative emphasis on reward components.
PHASE_EMPHASIS = [
    {"novelty": 1.5, "exploration": 1.3},
    {"compression": 1.2, "performance": 1.1},
    {"performance": 1.4, "stability": 1.2},
    {"compression": 1.5, "performance": 1.5},
]


class Curriculum:
    """Phase state machine driving the dynamic reward weighting."""

    def __init__(
        self,
        initial_param_target: float = 0.85,
        min_param_target: float = 0.15,
        thresholds=(0.3, 0.5, 0.7, 0.9),
        progress_window: int = 250,
        momentum: float = 0.9,
    ):
        self.current = 0
        self.thresholds = list(thresholds)
        self.param_target = initial_param_target
        self.min_param_target = min_param_target
        self.progress_window = progress_window
        self.momentum = momentum

    def update(self, performance_hist: Deque[float]) -> None:
        """Advance the phase when recent performance clears the threshold."""
        if len(performance_hist) < self.progress_window:
            return
        recent_perf = np.mean(list(performance_hist)[-self.progress_window:])
        target = self.thresholds[self.current]
        if recent_perf > target:
            self.current = min(self.current + 1, 3)
            self.param_target = max(self.param_target * 0.8, self.min_param_target)
            print(f"Curriculum: Advanced to Phase {self.current + 1}")

    @property
    def performance_config(self):
        return PHASE_PERFORMANCE_CONFIG[self.current]

    @property
    def emphasis(self) -> Dict[str, float]:
        return PHASE_EMPHASIS[self.current]

    # ------------------------------------------------------------------
    # (De)serialisation helpers for checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "current": self.current,
            "thresholds": self.thresholds,
            "param_target": self.param_target,
            "min_param_target": self.min_param_target,
            "progress_window": self.progress_window,
            "momentum": self.momentum,
        }

    def load_state_dict(self, state: Dict) -> None:
        for key, value in state.items():
            setattr(self, key, value)
