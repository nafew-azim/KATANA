"""Code-generation policy model (Sec. 3.5).

Loads the LLM proposer with 4-bit quantisation and attaches Low-Rank
Adapters, using an Unsloth-accelerated stack so the full generation and
GRPO update loop fits on a single A100 (40GB). The paper's discovery runs
use DeepSeek-Coder-V2-Lite-Instruct (16B) with LoRA r = 64, alpha = 128,
dropout 0.05; smaller proxy models (e.g. Phi-4, CodeLlama-7B) roughly halve
the search cost without significant degradation (Sec. 4.2).
"""

from typing import Optional

import torch


def load_policy(
    model_name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    max_seq_length: int = 2048,
    lora_rank: int = 64,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.05,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    gpu_memory_utilization: float = 0.7,
    dtype=torch.float16,
    random_state: int = 3407,
):
    """Return (model, tokenizer) ready for GRPO training.

    Imported lazily: Unsloth is a heavyweight, GPU-only dependency that the
    pruning/evaluation half of the package must not require.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=list(target_modules),
        lora_alpha=lora_alpha if lora_alpha is not None else lora_rank,
        lora_dropout=lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=random_state,
    )
    return model, tokenizer
