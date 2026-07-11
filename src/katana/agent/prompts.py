"""Structured prompts fed to the LLM proposer (Fig. 2).

``KATANA_SYSTEM_PROMPT`` is the exact generative system prompt template
provided to the policy model at each iteration of the discovery search.
It injects the current curriculum phase via ``{phase_description}`` and
enforces the ``<think>`` reasoning block prior to the executable code,
ensuring the model explicitly evaluates multimodal alignment before
outputting the Python Abstract Syntax Tree (AST).

The per-iteration task message carries the baseline pruning function
(e.g., Wanda) and historical performance metrics (Sec. 3.2).

During the search, the generated Python code is dynamically parsed from
the markdown block (see :func:`katana.agent.parser.parse_response`),
stripped of its comments and variable names to compute the structural AST
Novelty Reward, and then executed strictly within the restricted sandbox.
"""

import textwrap

# ----------------------------------------------------------------------
# Exact generative system prompt of the discovery search
# ----------------------------------------------------------------------
KATANA_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert AI researcher designing state-of-the-art neural
    network pruning algorithms for Vision-Language Models (VLMs). Your task
    is to write a Python function that determines which weights to drop to
    achieve a target sparsity s.

    Current Curriculum Phase: {phase_description}

    Inputs provided to your function:

    - weight: A 2D PyTorch tensor (out_features, in_features).
    - activations: A 1D tensor of mean activation magnitudes for the
      input channels.
    - sparsity: A float between 0.0 and 1.0 representing the target
      sparsity.
    - step, total_steps: Integers representing the current pruning step.

    Constraints:
    1. You must return a binary mask tensor of the exact same shape as
    weight, containing 1s (keep) and 0s (prune).
    2. You must achieve exactly the requested sparsity level globally.
    3. Consider the delicate alignment between vision and language
    modalities.
    4. Do not use external libraries other than torch and math.

    Output Format:
    First, provide your step-by-step mathematical reasoning inside <think>
    ... </think> tags. Explain WHY your metric will preserve VLM alignment.
    Then, provide the complete Python function inside a ```python ...
    ``` code block. The function MUST be named def prune_module(weight,
    activations, sparsity, step, total_steps):.
""").strip()

# Curriculum-phase descriptions injected into {phase_description},
# following the Fig. 2 curriculum timeline.
PHASE_DESCRIPTIONS = [
    "Phase 1 (Base Exploration): Focus on basic magnitude scoring.",
    "Phase 2 (Balancing): Incorporate activations.",
    "Phase 3 (Optimization): Add structural norms and reconstruction.",
    "Phase 4 (Aggressive Optimization): Jointly maximize compression and "
    "multimodal quality preservation.",
]


def build_system_prompt(phase: int = 0) -> str:
    """Fill the system prompt for the given curriculum phase."""
    return KATANA_SYSTEM_PROMPT.format(phase_description=PHASE_DESCRIPTIONS[phase])


# ----------------------------------------------------------------------
# Per-iteration task message (Sec. 3.2: baseline function + history)
# ----------------------------------------------------------------------
# Baseline pruning function included in the structured prompt (e.g., Wanda).
# Faithful to Wanda's per-output-row comparison groups; the activation term
# uses the mean absolute magnitudes supplied by the search contract in place
# of Wanda's L2 activation norm, which is not derivable from those inputs.
WANDA_BASELINE = textwrap.dedent("""
    def prune_module(weight, activations, sparsity, step, total_steps):
        # Baseline: Wanda-style one-shot pruning. Score = |W| * input-channel
        # activation magnitude, ranked within each output row.
        importance = weight.abs() * activations.unsqueeze(0)
        k = int(sparsity * weight.shape[1])
        threshold = importance.kthvalue(k, dim=1, keepdim=True).values
        return (importance > threshold).to(weight.dtype)
""").strip()


TASK_PROMPT = textwrap.dedent("""
    ## Target
    Prune the model to {target_sparsity:.0%} global unstructured sparsity while
    preserving task performance and reducing per-token decode latency.

    ## Baseline pruning function (improve upon this)
    ```python
    {baseline_code}
    ```

    ## Historical performance of your recent strategies
    {history}

    Design a new pruning strategy that advances beyond the baseline and your own
    history. Prioritize originality: novel importance formulations and novel
    sparsity trajectories score higher than variations of known heuristics.
    Remember the output format: <think> reasoning, then the ```python code block.
""").strip()


def build_task_prompt(
    target_sparsity: float = 0.7,
    baseline_code: str = WANDA_BASELINE,
    history: str = "No previous strategies evaluated yet.",
) -> str:
    """Fill the per-iteration task message for the discovery search."""
    return TASK_PROMPT.format(
        target_sparsity=target_sparsity,
        baseline_code=baseline_code,
        history=history,
    )


def build_grpo_dataset(task_prompt: str = None, system_prompt: str = None, phase: int = 0):
    """Wrap the prompts in the conversational format expected by GRPOTrainer."""
    if task_prompt is None:
        task_prompt = build_task_prompt()
    if system_prompt is None:
        system_prompt = build_system_prompt(phase)
    return [{
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
    }]
