#!/usr/bin/env python
"""Run the KATANA RL discovery search (Fig. 2).

An LLM proposer generates candidate pruning programs, each is validated and
measured inside the sandbox, and the policy is updated with GRPO under the
multi-objective reward. After training, a small batch of fresh candidates
is sampled and evaluated, and top-N leaderboards are reported.

Usage:
    python scripts/run_search.py --config configs/search.yaml
"""

import argparse
import warnings

import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description="KATANA RL pruning-strategy discovery")
    parser.add_argument("--config", default="configs/search.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from katana.utils.logging import setup_logging

    logger = setup_logging(cfg["logging"]["log_file"])

    # ------------------------------------------------------------------
    # Policy model (LLM proposer)
    # ------------------------------------------------------------------
    from katana.agent.policy import load_policy

    policy_cfg = cfg["policy"]
    model, tokenizer = load_policy(
        model_name=policy_cfg["model_name"],
        max_seq_length=policy_cfg["max_seq_length"],
        lora_rank=policy_cfg["lora_rank"],
        lora_alpha=policy_cfg.get("lora_alpha"),
        lora_dropout=policy_cfg.get("lora_dropout", 0.05),
        target_modules=policy_cfg["target_modules"],
        gpu_memory_utilization=policy_cfg["gpu_memory_utilization"],
        random_state=policy_cfg["random_state"],
    )

    # ------------------------------------------------------------------
    # Calibration model + dataset evaluated inside the sandbox
    # ------------------------------------------------------------------
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    calib_cfg = cfg["calibration"]
    model_to_prune = AutoModelForSequenceClassification.from_pretrained(calib_cfg["model_name"]).to(device)
    calib_tokenizer = AutoTokenizer.from_pretrained(calib_cfg["model_name"])
    val_dataset = load_dataset(calib_cfg["dataset"], calib_cfg["subset"], split=calib_cfg["split"]).select(
        range(calib_cfg["num_samples"])
    )

    # ------------------------------------------------------------------
    # Sandbox + multi-objective reward system
    # ------------------------------------------------------------------
    from katana.rewards.system import MultiObjectiveRewardSystem
    from katana.sandbox.executor import SandboxExecutor

    sandbox = SandboxExecutor(
        model_to_prune,
        calib_tokenizer,
        val_dataset,
        device,
        batch_size=calib_cfg["batch_size"],
        target_sparsity=cfg["search"]["target_sparsity"],
        exec_timeout=cfg["search"].get("exec_timeout", 60),
    )
    base_metrics = {
        "original_accuracy": sandbox.original_accuracy,
        "original_inference_time": sandbox.original_inference_time,
    }
    reward_cfg = cfg["reward"]
    reward_system = MultiObjectiveRewardSystem(
        base_metrics=base_metrics,
        strategy_capacity=reward_cfg["strategy_capacity"],
        initial_param_target=reward_cfg["initial_param_target"],
        top_n=reward_cfg["top_n"],
    )

    # ------------------------------------------------------------------
    # GRPO training
    # ------------------------------------------------------------------
    from katana.agent.prompts import build_grpo_dataset, build_task_prompt
    from katana.training.grpo import build_trainer
    from katana.training.reward_fn import make_pruning_reward_fn

    dataset = build_grpo_dataset(
        task_prompt=build_task_prompt(target_sparsity=cfg["search"]["target_sparsity"]),
        phase=0,
    )
    reward_fn = make_pruning_reward_fn(reward_system, sandbox)
    trainer = build_trainer(model, tokenizer, reward_fn, dataset, reward_system, base_metrics, cfg["grpo"])
    trainer.train()

    # ------------------------------------------------------------------
    # Final evaluation with freshly sampled candidates
    # ------------------------------------------------------------------
    logger.info("Training completed. Starting final evaluation with additional candidates.")
    from katana.agent.parser import parse_response
    from katana.training.callbacks import report_strategy, report_top_n
    from katana.training.reward_fn import STAGE_PENALTIES

    model.eval()
    final_cfg = cfg["final_evaluation"]
    prompt_text = "\n\n".join(m["content"] for m in dataset[0]["prompt"])

    for i in range(final_cfg["num_candidates"]):
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        generated_ids = model.generate(
            input_ids,
            max_length=final_cfg["max_length"],
            do_sample=True,
            top_p=final_cfg["top_p"],
            temperature=final_cfg["temperature"],
        )
        completion = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        reasoning, code = parse_response(completion)

        if not code:
            logger.info(f"Final candidate {i + 1} reward: -3.0 (No code provided)")
            continue

        try:
            result = sandbox.apply_pruning(code)
            if result.get("valid", False):
                reward = reward_system.calculate(result, code)
                reward_system.update_top_n(code, result, reasoning, reward, is_training=False)
                pruned = result["metrics"]["pruned"]
                logger.info(
                    f"Final candidate {i + 1} reward: {reward:.4f}, "
                    f"accuracy: {pruned['accuracy']:.4f}, "
                    f"inference_time: {pruned['inference_time']:.2f}s"
                )
                if reward > reward_system.best_reward:
                    reward_system.best_reward = reward
                    reward_system.best_candidate = code
                    reward_system.best_reasoning = reasoning
                    reward_system.best_result = result
                    logger.info(f"New best reward in final evaluation: {reward:.4f}")
            else:
                stage = result.get("stage", "unknown")
                reward = STAGE_PENALTIES.get(stage, -4.0)
                logger.info(f"Final candidate {i + 1} invalid: {stage}, reward: {reward:.4f}")
        except Exception as e:
            logger.error(f"Error in final candidate {i + 1}: {e}")

    # ------------------------------------------------------------------
    # Overall leaderboards + best strategy report
    # ------------------------------------------------------------------
    report_top_n("Overall Top N Candidates by Reward", reward_system.top_rewards, "Reward")
    report_top_n("Overall Top N Candidates by Accuracy", reward_system.top_accuracies, "Accuracy")
    report_top_n(
        "Overall Top N Candidates by Inference Time",
        reward_system.top_inference_times, "Inference Time", "{:.2f}s",
    )
    if reward_system.best_candidate:
        report_strategy(
            "Best Overall Pruning Strategy",
            reward_system.best_candidate, reward_system.best_reasoning,
            reward_system.best_result, base_metrics, best_reward=reward_system.best_reward,
        )
    else:
        logger.info("No valid pruning strategy found.")

    # ------------------------------------------------------------------
    # Save the fine-tuned proposer policy
    # ------------------------------------------------------------------
    save_path = cfg["save_path"]
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Saved policy to {save_path}")


if __name__ == "__main__":
    main()
