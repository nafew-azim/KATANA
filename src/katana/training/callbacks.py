"""Trainer callbacks: checkpointing, dashboards, and search reporting."""

import os
import textwrap

from transformers import TrainerCallback

from ..utils.logging import get_logger


def _log(msg: str) -> None:
    get_logger().info(msg)


# ----------------------------------------------------------------------
# Shared report formatters
# ----------------------------------------------------------------------
def report_strategy(title: str, code, reasoning, result, base_metrics, best_reward=None):
    """Log one strategy (code + reasoning + metrics) under a banner."""
    separator = "=" * 50
    _log(separator)
    _log(f"===== {title} =====")
    _log(f"Code:\n{textwrap.indent(code, '    ')}")
    _log("===== Reasoning =====")
    _log(textwrap.indent(reasoning, "    ") if reasoning else "No reasoning provided.")
    _log("===== Metrics =====")
    pruned = result["metrics"]["pruned"]
    _log(f"Accuracy: {pruned['accuracy']:.4f} (Original: {base_metrics['original_accuracy']:.4f})")
    _log(
        f"Inference Time: {pruned['inference_time']:.2f}s "
        f"(Original: {base_metrics['original_inference_time']:.2f}s)"
    )
    param_reduction = 1 - result["structural_changes"]["param_diff"]
    _log(f"Parameters Reduced: {param_reduction:.1%}")
    if best_reward is not None:
        _log(f"Best Reward: {best_reward:.4f}")
    _log(separator)


def report_top_n(title: str, entries, key_label: str, key_fmt: str = "{:.4f}"):
    """Log a top-N leaderboard of (key, code, result, reasoning) tuples."""
    _log(f"\n--- {title} ---")
    for i, (key, code, result, reasoning) in enumerate(entries):
        _log(f"Rank {i + 1}: {key_label} = {key_fmt.format(key)}")
        _log("Code:")
        _log(textwrap.indent(code, "    "))
        if reasoning:
            _log("Reasoning:")
            _log(textwrap.indent(reasoning, "    "))
        _log("Metrics:")
        pruned = result["metrics"]["pruned"]
        _log(f"Accuracy: {pruned['accuracy']:.4f}")
        _log(f"Inference Time: {pruned['inference_time']:.2f}s")
        param_reduction = 1 - result["structural_changes"]["param_diff"]
        _log(f"Parameters Reduced: {param_reduction:.1%}")
        _log("-" * 50)


# ----------------------------------------------------------------------
# Trainer callbacks: checkpoint persistence and live search reporting
# ----------------------------------------------------------------------
class SaveRewardSystemCallback(TrainerCallback):
    """Persist the reward-system state alongside each model checkpoint."""

    def __init__(self, reward_system):
        self.reward_system = reward_system

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        path = os.path.join(checkpoint_dir, "reward_system.pkl")
        self.reward_system.save_state(path)
        _log(f"Saved reward system state to {path}")


class DashboardCallback(TrainerCallback):
    """Print the curriculum/novelty dashboard every 10 steps."""

    def __init__(self, reward_system):
        self.reward_system = reward_system

    def on_log(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            _log(f"Displaying training dashboard at global step {state.global_step}")
            self.reward_system.training_dashboard()


class PrintCandidatesCallback(TrainerCallback):
    """Dump every candidate generated during the epoch, then clear the buffer."""

    def __init__(self, reward_system):
        self.reward_system = reward_system

    def on_epoch_end(self, args, state, control, **kwargs):
        _log(f"\n--- Candidates after Epoch {state.epoch} ---")
        if not self.reward_system.candidates_with_rewards:
            _log("No candidates generated in this epoch.")
        else:
            for i, (candidate, reward, reasoning, result) in enumerate(
                self.reward_system.candidates_with_rewards
            ):
                _log(f"Candidate {i + 1} (Reward: {reward:.4f}):")
                _log(textwrap.indent(candidate, "    "))
                if reasoning:
                    _log("Reasoning:")
                    _log(textwrap.indent(reasoning, "    "))
                if result and result.get("valid", False):
                    pruned = result["metrics"]["pruned"]
                    param_reduction = 1 - result["structural_changes"]["param_diff"]
                    _log(
                        f"Metrics: Accuracy={pruned['accuracy']:.4f}, "
                        f"Inference Time={pruned['inference_time']:.2f}s, "
                        f"Param Reduction={param_reduction:.1%}"
                    )
                else:
                    _log("Invalid pruning strategy.")
                _log("-" * 50)
        self.reward_system.candidates_with_rewards.clear()


class PrintRewardsCallback(TrainerCallback):
    """Log the mean reward of the epoch, then reset the accumulator."""

    def __init__(self, reward_system):
        self.reward_system = reward_system

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.reward_system.epoch_rewards:
            import numpy as np

            avg_reward = np.mean(self.reward_system.epoch_rewards)
            _log(f"Epoch {state.epoch} - Average Reward: {avg_reward:.4f}")
            self.reward_system.epoch_rewards.clear()


class SearchSummaryCallback(TrainerCallback):
    """Final end-of-training report: best strategies per objective + top-N."""

    def __init__(self, reward_system, base_metrics):
        self.reward_system = reward_system
        self.base_metrics = base_metrics

    def on_train_end(self, args, state, control, **kwargs):
        rs = self.reward_system
        if rs.best_candidate:
            report_strategy(
                "Best Pruning Strategy by Reward",
                rs.best_candidate, rs.best_reasoning, rs.best_result,
                self.base_metrics, best_reward=rs.best_reward,
            )
        if rs.best_accuracy_candidate:
            report_strategy(
                "Best Pruning Strategy by Accuracy",
                rs.best_accuracy_candidate, rs.best_accuracy_reasoning,
                rs.best_accuracy_result, self.base_metrics,
            )
        if rs.best_inference_time_candidate:
            report_strategy(
                "Best Pruning Strategy by Inference Time",
                rs.best_inference_time_candidate, rs.best_inference_time_reasoning,
                rs.best_inference_time_result, self.base_metrics,
            )
        if not (rs.best_candidate or rs.best_accuracy_candidate or rs.best_inference_time_candidate):
            _log("No valid pruning strategy found.")

        report_top_n("Top N Candidates from Training by Reward", rs.training_top_rewards, "Reward")
        report_top_n("Top N Candidates from Training by Accuracy", rs.training_top_accuracies, "Accuracy")
        report_top_n(
            "Top N Candidates from Training by Inference Time",
            rs.training_top_inference_times, "Inference Time", "{:.2f}s",
        )
