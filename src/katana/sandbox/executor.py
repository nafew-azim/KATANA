"""Sandbox Execution Environment (Sec. 3.2, Fig. 2).

Every candidate pruning program proposed by the LLM agent passes through a
strict validation pipeline before it may touch the calibration model:

1. Security screening — static pattern rejection (``security.py``).
2. Restricted compilation — the candidate is compiled with
   RestrictedPython, rejecting disallowed constructs at the AST level.
3. Quarantined execution on dummy tensors — the program runs under
   RestrictedPython guards with a minimal namespace (``torch``, ``nn``,
   ``math``, ``copy``) and its ``prune_module(weight, activations,
   sparsity, step, total_steps)`` is first exercised on small dummy
   tensors before any full model execution. A wall-clock timeout bounds
   the whole evaluation.
4. Model masking + validation — the masks are applied to every
   eligible linear layer of a deep copy of the calibration model, which
   must still complete a forward pass.
5. Measurement — accuracy, inference time, achieved sparsity, and the
   per-layer sparsity distribution are collected for the multi-objective
   reward.

Each failure mode maps to a distinct ``stage`` (security / code_validation /
execution / model_validation / timeout) so the reward system can penalise
precisely.
"""

import copy
import math
import operator
import signal
import textwrap
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
)

from ..utils.logging import get_logger
from .security import contains_dangerous_code

# Layer-name substrings exempted from pruning (embeddings, output heads).
DEFAULT_EXCLUDE_PATTERNS = ("embed", "lm_head", "head", "classifier", "pooler")


def _log(msg: str) -> None:
    get_logger().info(msg)


def _inplacevar(op, x, y):
    ops = {
        "+=": operator.iadd, "-=": operator.isub, "*=": operator.imul,
        "/=": operator.itruediv, "//=": operator.ifloordiv,
        "%=": operator.imod, "**=": operator.ipow,
    }
    return ops[op](x, y)


def _restricted_namespace() -> Dict[str, Any]:
    """RestrictedPython globals exposing only the contract's allowed modules."""
    return {
        "__builtins__": dict(safe_builtins),
        "torch": torch,
        "nn": nn,
        "math": math,
        "copy": copy,
        "_getattr_": safer_getattr,
        "_getitem_": lambda obj, key: obj[key],
        "_getiter_": iter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_write_": lambda obj: obj,
        "_inplacevar_": _inplacevar,
    }


@contextmanager
def _time_limit(seconds: int):
    """SIGALRM-based wall-clock limit (Unix main thread; no-op elsewhere)."""
    usable = (
        seconds
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not usable:
        yield
        return

    def _raise(signum, frame):
        raise TimeoutError("candidate execution timed out")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class SandboxExecutor:
    """Owns the calibration model and safely evaluates candidate programs."""

    def __init__(
        self,
        model,
        tokenizer,
        val_dataset,
        device: str = "cuda",
        batch_size: int = 16,
        target_sparsity: float = 0.7,
        exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
        exec_timeout: int = 60,
    ):
        self.original_model = copy.deepcopy(model)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.target_sparsity = target_sparsity
        self.exclude_patterns = tuple(exclude_patterns)
        self.exec_timeout = exec_timeout

        # Pre-tokenise the evaluation set once; every candidate is then
        # measured on identical tensors for a fair comparison.
        self.pre_tokenized = []
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        for batch in self.val_loader:
            inputs = tokenizer(
                batch["sentence1"],
                batch.get("sentence2", None),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor(batch["label"]).to(device)
            self.pre_tokenized.append((inputs, labels))

        self.eligible_layers = self._eligible_layers(self.model)
        self.activation_profiles = self._calibrate_activations()
        self.original_accuracy, self.original_inference_time = self._compute_metrics(self.original_model)

    # ------------------------------------------------------------------
    # Layer eligibility and activation calibration (candidate inputs)
    # ------------------------------------------------------------------
    def _eligible_layers(self, model) -> Dict[str, nn.Linear]:
        layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not any(
                pat in name.lower() for pat in self.exclude_patterns
            ):
                layers[name] = module
        return layers

    def _calibrate_activations(self) -> Dict[str, torch.Tensor]:
        """Mean absolute input activation per channel, per eligible layer.

        These are the ``activations`` handed to every candidate's
        ``prune_module``, aggregated over the calibration batches.
        """
        profiles, counts, handles = {}, {}, []

        def make_hook(name):
            def hook(module, inputs, output):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                batch_mean = x.abs().float().mean(dim=0)
                n = x.shape[0]
                if name in profiles:
                    total = counts[name] + n
                    profiles[name] = profiles[name] * (counts[name] / total) + batch_mean * (n / total)
                    counts[name] = total
                else:
                    profiles[name], counts[name] = batch_mean, n
            return hook

        for name, layer in self.eligible_layers.items():
            handles.append(layer.register_forward_hook(make_hook(name)))
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in self.pre_tokenized:
                self.model(**inputs)
        for handle in handles:
            handle.remove()
        return profiles

    # ------------------------------------------------------------------
    # Accuracy and latency measurement on the fixed evaluation set
    # ------------------------------------------------------------------
    def _compute_metrics(self, model):
        model.eval()
        correct, total = 0, 0
        start_time = time.time()
        with torch.no_grad():
            for inputs, labels in self.pre_tokenized:
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        inference_time = time.time() - start_time
        return accuracy, inference_time

    def _validate_model(self, model) -> bool:
        try:
            test_input = self.tokenizer("Test input", return_tensors="pt").to(self.device)
            model(**test_input)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Candidate evaluation pipeline
    # ------------------------------------------------------------------
    def apply_pruning(self, code_str: str) -> Dict[str, Any]:
        result = {
            "valid": False,
            "errors": [],
            "stage": "unknown",
            "metrics": {
                "original": {
                    "accuracy": self.original_accuracy,
                    "inference_time": self.original_inference_time,
                },
                "pruned": {"accuracy": 0.0, "inference_time": 0.0},
            },
            "structural_changes": {"structural_diff": [], "param_diff": 1.0, "layer_changes": {}},
        }

        # Stage 1: security screening
        if contains_dangerous_code(code_str):
            result.update({"errors": ["Security violation detected"], "stage": "security"})
            _log(f"Result: {result}")
            return result

        # Stage 2: restricted compilation (syntax + disallowed constructs)
        try:
            code_str = textwrap.dedent(code_str).strip()
            byte_code = compile_restricted(code_str, "<candidate>", "exec")
        except SyntaxError as e:
            result.update({"errors": [f"Syntax error: {e}"], "stage": "code_validation"})
            _log(f"Result: {result}")
            return result

        try:
            with _time_limit(self.exec_timeout):
                # Stage 3: quarantined execution + dummy-tensor smoke test
                local_vars = {}
                exec(byte_code, _restricted_namespace(), local_vars)  # noqa: S102 — RestrictedPython bytecode

                prune_fn = local_vars.get("prune_module")
                if not callable(prune_fn):
                    result.update({"errors": ["Missing valid prune_module function"], "stage": "code_validation"})
                    _log(f"Result: {result}")
                    return result

                dummy_weight = torch.randn(8, 16)
                dummy_mask = prune_fn(dummy_weight, torch.rand(16), self.target_sparsity, 1, 1)
                self._validate_mask(dummy_mask, dummy_weight, "<dummy>")

                # Stage 4: apply masks to every eligible layer of a deep copy
                model_copy = copy.deepcopy(self.model)
                copy_layers = self._eligible_layers(model_copy)
                layer_changes, zeros, total = {}, 0, 0
                with torch.no_grad():
                    for name, layer in copy_layers.items():
                        weight = layer.weight.data
                        activations = self.activation_profiles[name].to(weight.device, weight.dtype)
                        mask = prune_fn(weight, activations, self.target_sparsity, 1, 1)
                        self._validate_mask(mask, weight, name)
                        mask = mask.to(weight.device, weight.dtype)
                        layer.weight.data = weight * mask
                        layer_zeros = int((mask == 0).sum())
                        layer_changes[name] = layer_zeros / mask.numel()
                        zeros += layer_zeros
                        total += mask.numel()

            if not self._validate_model(model_copy):
                result.update({"errors": ["Pruned model failed validation"], "stage": "model_validation"})
                _log(f"Result: {result}")
                return result

            # Stage 5: measurement
            pruned_accuracy, pruned_inference_time = self._compute_metrics(model_copy)
            achieved_sparsity = zeros / max(total, 1)
            param_diff = 1.0 - achieved_sparsity  # fraction of weights retained
            structural_diff = [f"{name}: {sparsity:.1%}" for name, sparsity in layer_changes.items()]

            result.update({
                "valid": True,
                "stage": "success",
                "metrics": {
                    "original": result["metrics"]["original"],
                    "pruned": {"accuracy": pruned_accuracy, "inference_time": pruned_inference_time},
                },
                "structural_changes": {
                    "structural_diff": structural_diff,
                    "param_diff": param_diff,
                    "layer_changes": layer_changes,
                },
            })

        except TimeoutError as e:
            result.update({"errors": [str(e)], "stage": "timeout"})
        except ValueError as e:
            result.update({"errors": [f"Configuration error: {e}"], "stage": "execution"})
        except Exception as e:
            result.update({"errors": [f"Execution error: {e}"], "stage": "execution"})

        _log(f"Result: {result}")
        return result

    @staticmethod
    def _validate_mask(mask, weight, name: str) -> None:
        if not torch.is_tensor(mask) or mask.shape != weight.shape:
            raise ValueError(f"prune_module returned an invalid mask for layer {name}")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"mask for layer {name} is not binary")
