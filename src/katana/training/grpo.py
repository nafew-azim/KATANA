"""GRPO trainer assembly (Sec. 3.2, third stage).

KATANA does not rely solely on evolutionary prompt feedback: the agent's
policy is updated via gradients derived from the multi-objective reward
using Group Relative Policy Optimization. An Evolutionary Search baseline
plateaus (78.4 CIDEr) due to mutation instability, whereas GRPO safely
navigates the reward landscape (82.5 CIDEr at 300 iterations).
"""

from trl import GRPOConfig, GRPOTrainer

from .callbacks import (
    DashboardCallback,
    PrintCandidatesCallback,
    PrintRewardsCallback,
    SaveRewardSystemCallback,
    SearchSummaryCallback,
)


def build_trainer(model, tokenizer, reward_fn, dataset, reward_system, base_metrics, grpo_kwargs):
    """Wire the policy, reward function, and callbacks into a GRPOTrainer."""
    training_args = GRPOConfig(**grpo_kwargs)
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        callbacks=[
            SaveRewardSystemCallback(reward_system),
            DashboardCallback(reward_system),
            PrintCandidatesCallback(reward_system),
            PrintRewardsCallback(reward_system),
            SearchSummaryCallback(reward_system, base_metrics),
        ],
    )
