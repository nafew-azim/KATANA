"""GRPO reward function bridging completions -> sandbox -> reward system."""

import sys
import traceback

from ..agent.parser import parse_response
from ..utils.logging import get_logger

# Staged penalties for invalid candidates (Fig. 2 failure taxonomy).
STAGE_PENALTIES = {"security": -5.0, "code_validation": -4.75, "execution": -4.5}
NO_CODE_PENALTY = -3.0
REASONING_BONUS = 0.5  # granted for a substantive <think> rationale (> 20 words)


def make_pruning_reward_fn(reward_system, sandbox):
    """Build the reward callable consumed by ``GRPOTrainer``.

    For every completion: parse the ``<think>``/``<code>`` sections, run the
    candidate through the sandbox pipeline, synthesise the multi-objective
    reward, and update best-so-far / top-N leaderboards.
    """
    logger = get_logger()

    def pruning_reward_func(prompts, completions, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            reward_system.candidates.append(response)
            reasoning, code = parse_response(response)
            reward = None
            result = None

            if not code:
                reward = NO_CODE_PENALTY
                scores.append(reward)
                reward_system.candidates_with_rewards.append((response, reward, reasoning, result))
                continue

            try:
                result = sandbox.apply_pruning(code)
                if not result.get("valid", False):
                    stage = result.get("stage", "unknown")
                    reward = STAGE_PENALTIES.get(stage, -4.0)
                    scores.append(reward)
                    reward_system.candidates_with_rewards.append((response, reward, reasoning, result))
                else:
                    reward = reward_system.calculate(result, code)
                    accuracy = result["metrics"]["pruned"]["accuracy"]
                    inference_time = result["metrics"]["pruned"]["inference_time"]

                    if reasoning and len(reasoning.split()) > 20:
                        reward += REASONING_BONUS

                    scores.append(reward)
                    reward_system.candidates_with_rewards.append((response, reward, reasoning, result))

                    if reward > reward_system.best_reward:
                        reward_system.best_reward = reward
                        reward_system.best_candidate = code
                        reward_system.best_reasoning = reasoning
                        reward_system.best_result = result
                        logger.info(f"New best reward found: {reward:.4f}")
                        logger.info(f"Best reasoning:\n{reasoning}")
                        logger.info(f"Best candidate:\n{code}")

                    if accuracy > reward_system.best_accuracy:
                        reward_system.best_accuracy = accuracy
                        reward_system.best_accuracy_candidate = code
                        reward_system.best_accuracy_result = result
                        reward_system.best_accuracy_reasoning = reasoning
                        logger.info(f"New best accuracy found: {accuracy:.4f}")

                    if inference_time < reward_system.best_inference_time:
                        reward_system.best_inference_time = inference_time
                        reward_system.best_inference_time_candidate = code
                        reward_system.best_inference_time_result = result
                        reward_system.best_inference_time_reasoning = reasoning
                        logger.info(f"New best inference time found: {inference_time:.2f}s")

                    reward_system.update_top_n(code, result, reasoning, reward, is_training=True)

            except SyntaxError:
                reward = -4.0
                scores.append(reward)
                reward_system.candidates_with_rewards.append((response, reward, reasoning, result))
            except Exception as e:
                print(f"Unexpected error: {e}", file=sys.stderr)
                logger.error(f"Unexpected error: {e}")
                traceback.print_exc()
                reward = -3.0
                scores.append(reward)
                reward_system.candidates_with_rewards.append((response, reward, reasoning, result))

        reward_system.epoch_rewards.extend(scores)
        return scores

    return pruning_reward_func
