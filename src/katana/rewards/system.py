"""Multi-objective reward system (Sec. 3.3).

Aggregates six reward components with adaptive weights,
R = sum_k w_k * r_k:

* performance (r_p) — phase-aware blend of bounded accuracy and
  latency sigmoids (Eq. 3), giving smooth non-vanishing gradients for GRPO.
* compression (r_c) — momentum-augmented: a -4.0 penalty below 5%
  reduction, linear scaling toward the target, and a logarithmic bonus
  for exceeding it.
* novelty (r_n) — four-dimensional similarity index (structure,
  parametric behaviour, layer targeting, techniques) against a memory
  buffer of up to 10,000 strategy signatures. To defeat false novelty via
  trivial renaming, candidate scripts are parsed into ASTs and stripped
  of superficial identifiers before the Ratcliff-Obershelp similarity is
  computed.
* exploration — decaying bonus for touching rarely-engaged layer types
  and techniques.
* diversity (r_d) — long-term population diversity via sampled
  signature comparison.
* stability (r_s) — discrete bonus when accuracy improves
  monotonically over the five most recent generations.

Dynamic weighting through the adaptive curriculum prevents premature
convergence: low novelty (< 0.25) automatically raises the novelty weight,
while high novelty (> 0.60) paired with low performance lowers it.
"""

import ast
import difflib
import pickle
import re
import textwrap
from collections import defaultdict, deque
from typing import Any, Dict, Optional

import numpy as np

from .curriculum import Curriculum


class _IdentifierStripper(ast.NodeTransformer):
    """Anonymise function/variable names so trivially renamed scripts collide."""

    def visit_FunctionDef(self, node):
        node.name = "_"
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        node.name = "_"
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        node.id = "_"
        return node

    def visit_arg(self, node):
        node.arg = "_"
        return node


def normalize_code_ast(code: str) -> str:
    """AST dump of ``code`` with superficial identifiers stripped.

    Attribute names (e.g. the torch API surface) are preserved, since they
    carry the strategy's actual behaviour; comments vanish with the parse.
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return code
    return ast.dump(_IdentifierStripper().visit(tree))


FAILURE_PENALTIES = {
    "security": -5.0,
    "validation": -3.0,
    "execution": -2.0,
    "timeout": -1.5,
    "unknown": -1.0,
}


class MultiObjectiveRewardSystem:
    """Synthesises the reward signal that drives the GRPO policy update."""

    def __init__(
        self,
        base_metrics: Dict[str, Any],
        strategy_capacity: int = 10000,
        initial_param_target: float = 0.85,
        top_n: int = 3,
    ):
        self.base_acc = base_metrics["original_accuracy"]
        self.base_inf_time = base_metrics["original_inference_time"]
        self.curriculum = Curriculum(initial_param_target=initial_param_target)

        self.strategy_memory = deque(maxlen=strategy_capacity)
        self.layer_engagement = defaultdict(int)
        self.technique_usage = defaultdict(int)
        self.performance_hist = deque(maxlen=1000)
        self.accuracy_hist = deque(maxlen=1000)
        self.novelty_hist = deque(maxlen=500)

        self.dynamic_weights = {
            "compression": 0.5,
            "performance": 0.5,
            "novelty": 0.7,
            "exploration": 0.6,
            "diversity": 0.4,
            "stability": 0.3,
        }

        # Candidate bookkeeping
        self.candidates = []
        self.candidates_with_rewards = []  # (candidate, reward, reasoning, result)
        self.epoch_rewards = []

        # Best-so-far tracking along each objective
        self.best_reward = -np.inf
        self.best_candidate = None
        self.best_result = None
        self.best_reasoning = None
        self.best_accuracy = -1
        self.best_accuracy_candidate = None
        self.best_accuracy_result = None
        self.best_accuracy_reasoning = None
        self.best_inference_time = float("inf")
        self.best_inference_time_candidate = None
        self.best_inference_time_result = None
        self.best_inference_time_reasoning = None

        # Top-N leaderboards (overall and training-only)
        self.top_n = top_n
        self.top_rewards = []
        self.top_accuracies = []
        self.top_inference_times = []
        self.training_top_rewards = []
        self.training_top_accuracies = []
        self.training_top_inference_times = []

    # ------------------------------------------------------------------
    # Entry points called by the GRPO training loop
    # ------------------------------------------------------------------
    def calculate(self, eval_result: Dict[str, Any], code: Optional[str] = None) -> float:
        """Reward for one evaluated candidate. Failures get staged penalties."""
        if not eval_result.get("valid", False):
            return self._failure_penalty(eval_result)

        reward = self._compute_hybrid_reward(eval_result, code)
        pruned_metrics = eval_result["metrics"]["pruned"]

        # Hard guards against reward hacking
        if pruned_metrics["accuracy"] < 0.95 * self.base_acc:
            print("[DEBUG] Pruned accuracy below 95% of original. Applying penalty.")
            reward -= 2.0
        if pruned_metrics["inference_time"] > 1.1 * self.base_inf_time:
            print("[DEBUG] Pruned inference time above 110% of original. Applying penalty.")
            reward -= 2.0
        if code and re.search(r"return model\s*$", code):
            print("[DEBUG] Placeholder implementation detected. Applying heavy penalty.")
            reward -= 8.0

        return np.clip(reward, -3.0, 6.0)

    def update_top_n(self, code, result, reasoning, reward, is_training: bool = False):
        if not result.get("valid", False):
            return

        self.top_rewards.append((reward, code, result, reasoning))
        self.top_rewards.sort(key=lambda x: x[0], reverse=True)
        self.top_rewards = self.top_rewards[: self.top_n]

        accuracy = result["metrics"]["pruned"]["accuracy"]
        self.top_accuracies.append((accuracy, code, result, reasoning))
        self.top_accuracies.sort(key=lambda x: x[0], reverse=True)
        self.top_accuracies = self.top_accuracies[: self.top_n]

        inference_time = result["metrics"]["pruned"]["inference_time"]
        self.top_inference_times.append((inference_time, code, result, reasoning))
        self.top_inference_times.sort(key=lambda x: x[0])  # ascending: fastest first
        self.top_inference_times = self.top_inference_times[: self.top_n]

        if is_training:
            self.training_top_rewards.append((reward, code, result, reasoning))
            self.training_top_rewards.sort(key=lambda x: x[0], reverse=True)
            self.training_top_rewards = self.training_top_rewards[: self.top_n]

            self.training_top_accuracies.append((accuracy, code, result, reasoning))
            self.training_top_accuracies.sort(key=lambda x: x[0], reverse=True)
            self.training_top_accuracies = self.training_top_accuracies[: self.top_n]

            self.training_top_inference_times.append((inference_time, code, result, reasoning))
            self.training_top_inference_times.sort(key=lambda x: x[0])
            self.training_top_inference_times = self.training_top_inference_times[: self.top_n]

    # ------------------------------------------------------------------
    # Reward synthesis
    # ------------------------------------------------------------------
    def _compute_hybrid_reward(self, result: Dict[str, Any], code: Optional[str] = None) -> float:
        structural = dict(result["structural_changes"])
        if code:
            # Identifier-stripped AST: the 'structure' novelty dimension.
            structural["code_ast"] = normalize_code_ast(code)
        metrics = result["metrics"]["pruned"]
        param_diff = structural.get("param_diff", 1.0)

        self.curriculum.update(self.performance_hist)

        reward_components = {
            "compression": self._momentum_compression(param_diff),
            "performance": self._phase_aware_performance(metrics),
            "novelty": self._quad_novelty(structural),
            "exploration": self._decaying_exploration(structural),
            "diversity": self._longterm_diversity(structural),
            "stability": self._stability_bonus(),
        }

        self._adaptive_weighting()
        phase_emphasis = self.curriculum.emphasis

        total_reward = sum(
            self.dynamic_weights[k] * reward_components[k] * phase_emphasis.get(k, 1.0)
            for k in reward_components
        )

        self._update_tracking(structural, metrics)
        return np.clip(total_reward, -3.0, 6.0)

    def _momentum_compression(self, param_diff: float) -> float:
        reduction = 1 - param_diff
        target = 1 - self.curriculum.param_target
        momentum = 1 + (self.curriculum.momentum * len(self.performance_hist) / 1000)
        if reduction < 0.05:
            return -4.0
        elif reduction < target:
            return 3.5 * momentum * (reduction / target)
        # Logarithmic bonus for exceeding the target (caps at 7.0).
        excess = min((reduction - target) / (1 - target + 1e-8), 1.0)
        return 4.0 + 3.0 * np.log1p(9.0 * excess) / np.log(10.0)

    def _phase_aware_performance(self, metrics: Dict) -> float:
        """Eq. 3: r_p = a_phi * 3.0/(1+e^{-8(acc_ratio - tau_phi)})
        + b_phi * 2.5/(1+e^{-10(1.2 - latency_ratio)})."""
        acc_ratio = metrics["accuracy"] / self.base_acc
        latency_ratio = metrics["inference_time"] / self.base_inf_time
        acc_weight, speed_weight, tau = self.curriculum.performance_config
        acc_reward = 3.0 / (1 + np.exp(-8 * (acc_ratio - tau)))
        speed_reward = 2.5 / (1 + np.exp(-10 * (1.2 - latency_ratio)))
        return acc_weight * acc_reward + speed_weight * speed_reward

    # ------------------------------------------------------------------
    # Novelty / diversity / exploration / stability
    # ------------------------------------------------------------------
    def _strategy_signature(self, structural: Dict) -> Dict:
        # 'structure' prefers the identifier-stripped AST of the candidate
        # script; the model's structural diff is the fallback.
        return {
            "structure": str(structural.get("code_ast") or structural.get("structural_diff", [])),
            "param": structural.get("param_diff", 1.0),
            "layers": set(structural.get("layer_changes", {}).keys()),
            "tech": set(["pruning"]),
        }

    def _strategy_similarity(self, a: Dict, b: Dict) -> float:
        return (
            0.3 * difflib.SequenceMatcher(None, a["structure"], b["structure"]).ratio()
            + 0.25 * (1 - abs(a["param"] - b["param"]))
            + 0.25 * len(a["layers"] & b["layers"]) / (len(a["layers"] | b["layers"]) + 1e-8)
            + 0.2 * len(a["tech"] & b["tech"]) / (len(a["tech"] | b["tech"]) + 1e-8)
        )

    def _quad_novelty(self, structural: Dict) -> float:
        if not self.strategy_memory:
            return 5.0
        current_sig = self._strategy_signature(structural)
        similarities = [
            self._strategy_similarity(current_sig, self._strategy_signature(strategy))
            for strategy in self.strategy_memory
        ]
        return 4.0 * (1 - np.max(similarities)) if similarities else 3.0

    def _decaying_exploration(self, structural: Dict) -> float:
        bonus = 0.0
        decay = 1.0 / (1 + 0.005 * len(self.strategy_memory))
        for layer in structural.get("layer_changes", {}):
            layer_type = ".".join(layer.split(".")[:3])
            engagement = self.layer_engagement[layer_type]
            bonus += 0.7 * decay / (1 + np.sqrt(engagement))
        for tech in structural.get("pruning_techniques", []):
            usage = self.technique_usage[tech]
            bonus += 0.5 * decay / (1 + np.sqrt(usage))
        return np.tanh(bonus * 3)

    def _longterm_diversity(self, structural: Dict) -> float:
        if len(self.strategy_memory) < 100:
            return 0.5
        current_sig = self._strategy_signature(structural)
        sampled = [s for s in self.strategy_memory if np.random.rand() < 0.1]
        sampled_signatures = [self._strategy_signature(s) for s in sampled]
        similarities = [self._strategy_similarity(current_sig, s_sig) for s_sig in sampled_signatures]
        return 0.8 * (1 - np.max(similarities)) if similarities else 0.5

    def _stability_bonus(self) -> float:
        """Discrete bonus for monotonic accuracy gains over 5 generations."""
        if len(self.accuracy_hist) < 5:
            return 0.0
        recent = list(self.accuracy_hist)[-5:]
        return 0.5 if all(x < y for x, y in zip(recent, recent[1:])) else 0.0

    def _adaptive_weighting(self):
        if len(self.novelty_hist) > 100:
            recent_novelty = np.mean(list(self.novelty_hist)[-100:])
            if recent_novelty < 0.25:
                self.dynamic_weights["novelty"] = min(0.9, self.dynamic_weights["novelty"] + 0.05)
                self.dynamic_weights["compression"] = max(0.2, self.dynamic_weights["compression"] - 0.05)
            elif recent_novelty > 0.6 and len(self.performance_hist) > 200:
                recent_perf = np.mean(list(self.performance_hist)[-200:])
                if recent_perf < 0.1:
                    self.dynamic_weights["performance"] = min(0.8, self.dynamic_weights["performance"] + 0.1)
                    self.dynamic_weights["novelty"] = max(0.3, self.dynamic_weights["novelty"] - 0.1)

    def _update_tracking(self, structural: Dict, metrics: Dict):
        # Novelty is logged against the memory *before* this strategy joins
        # it, otherwise self-similarity pins every logged value to zero.
        self.novelty_hist.append(self._quad_novelty(structural))
        self.strategy_memory.append(structural)
        self.performance_hist.append(1 - structural.get("param_diff", 1.0))
        self.accuracy_hist.append(metrics.get("accuracy", 0.0))
        for layer in structural.get("layer_changes", {}):
            layer_type = ".".join(layer.split(".")[:3])
            self.layer_engagement[layer_type] += 1
        for tech in structural.get("pruning_techniques", []):
            self.technique_usage[tech] += 1

    def _failure_penalty(self, result: Dict) -> float:
        return FAILURE_PENALTIES.get(result.get("stage", "unknown"), -1.0)

    # ------------------------------------------------------------------
    # Dashboard and persistence
    # ------------------------------------------------------------------
    def training_dashboard(self):
        print("\n--- Training Dashboard ---")
        print(f"Current Phase: {self.curriculum.current + 1}")
        print(f"Param Target: {1 - self.curriculum.param_target:.1%}")
        if len(self.performance_hist) > 0:
            print(f"Recent Performance: {np.mean(list(self.performance_hist)[-100:]):.2f}")
        if len(self.novelty_hist) > 0:
            print(f"Novelty Level: {np.mean(list(self.novelty_hist)[-100:]):.2f}")
        print(f"Layer Engagement: {len(self.layer_engagement)} unique layers")
        print(f"Technique Diversity: {len(self.technique_usage)} methods")

    def save_state(self, filepath: str):
        state = {
            "base_acc": self.base_acc,
            "base_inf_time": self.base_inf_time,
            "curriculum": self.curriculum.state_dict(),
            "strategy_memory": list(self.strategy_memory),
            "layer_engagement": dict(self.layer_engagement),
            "technique_usage": dict(self.technique_usage),
            "performance_hist": list(self.performance_hist),
            "accuracy_hist": list(self.accuracy_hist),
            "novelty_hist": list(self.novelty_hist),
            "dynamic_weights": self.dynamic_weights,
            "candidates": self.candidates,
            "candidates_with_rewards": self.candidates_with_rewards,
            "best_reward": self.best_reward,
            "best_candidate": self.best_candidate,
            "best_result": self.best_result,
            "best_reasoning": self.best_reasoning,
            "best_accuracy": self.best_accuracy,
            "best_accuracy_candidate": self.best_accuracy_candidate,
            "best_accuracy_result": self.best_accuracy_result,
            "best_accuracy_reasoning": self.best_accuracy_reasoning,
            "best_inference_time": self.best_inference_time,
            "best_inference_time_candidate": self.best_inference_time_candidate,
            "best_inference_time_result": self.best_inference_time_result,
            "best_inference_time_reasoning": self.best_inference_time_reasoning,
            "top_rewards": self.top_rewards,
            "top_accuracies": self.top_accuracies,
            "top_inference_times": self.top_inference_times,
            "training_top_rewards": self.training_top_rewards,
            "training_top_accuracies": self.training_top_accuracies,
            "training_top_inference_times": self.training_top_inference_times,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        # Safety: checkpoints are written exclusively by save_state() on the
        # user's own machine; only load reward-system checkpoints you created.
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.base_acc = state["base_acc"]
        self.base_inf_time = state["base_inf_time"]
        self.curriculum.load_state_dict(state["curriculum"])
        self.strategy_memory = deque(state["strategy_memory"], maxlen=self.strategy_memory.maxlen)
        self.layer_engagement = defaultdict(int, state["layer_engagement"])
        self.technique_usage = defaultdict(int, state["technique_usage"])
        self.performance_hist = deque(state["performance_hist"], maxlen=self.performance_hist.maxlen)
        self.accuracy_hist = deque(state.get("accuracy_hist", []), maxlen=self.accuracy_hist.maxlen)
        self.novelty_hist = deque(state["novelty_hist"], maxlen=self.novelty_hist.maxlen)
        self.dynamic_weights = state["dynamic_weights"]
        self.candidates = state.get("candidates", [])
        self.candidates_with_rewards = state.get("candidates_with_rewards", [])
        self.best_reward = state.get("best_reward", -np.inf)
        self.best_candidate = state.get("best_candidate")
        self.best_result = state.get("best_result")
        self.best_reasoning = state.get("best_reasoning")
        self.best_accuracy = state.get("best_accuracy", -1)
        self.best_accuracy_candidate = state.get("best_accuracy_candidate")
        self.best_accuracy_result = state.get("best_accuracy_result")
        self.best_accuracy_reasoning = state.get("best_accuracy_reasoning")
        self.best_inference_time = state.get("best_inference_time", float("inf"))
        self.best_inference_time_candidate = state.get("best_inference_time_candidate")
        self.best_inference_time_result = state.get("best_inference_time_result")
        self.best_inference_time_reasoning = state.get("best_inference_time_reasoning")
        self.top_rewards = state.get("top_rewards", [])
        self.top_accuracies = state.get("top_accuracies", [])
        self.top_inference_times = state.get("top_inference_times", [])
        self.training_top_rewards = state.get("training_top_rewards", [])
        self.training_top_accuracies = state.get("training_top_accuracies", [])
        self.training_top_inference_times = state.get("training_top_inference_times", [])
