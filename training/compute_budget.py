"""Chinchilla-aware epoch reporting for POEM.

The 20 tokens/parameter reference is the Hoffmann et al. (2022) Chinchilla
rule of thumb.  It is useful as a scale check, not an exact optimum for small
symbolic-music CPU runs.  Training itself uses an explicit epoch count so every
candidate sees the same number of passes over the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


CHINCHILLA_TOKENS_PER_PARAM = 20.0
DEFAULT_TRAIN_EPOCHS = 40


@dataclass(frozen=True)
class ComputeBudgetPlan:
    dataset_tokens: int
    param_count: int
    tokens_per_param_per_epoch: float
    reference_tokens_per_param: float
    epochs_to_reference: int
    selected_epochs: int
    selected_tokens_per_param: float
    warning: str | None


def plan_epoch_budget(
    dataset_tokens: int,
    param_count: int,
    selected_epochs: int = DEFAULT_TRAIN_EPOCHS,
) -> ComputeBudgetPlan:
    if dataset_tokens <= 0:
        raise ValueError("dataset_tokens must be positive")
    if param_count <= 0:
        raise ValueError("param_count must be positive")
    if selected_epochs <= 0:
        raise ValueError("selected_epochs must be positive")
    tokens_per_param = dataset_tokens / param_count
    epochs_to_reference = max(1, math.ceil(CHINCHILLA_TOKENS_PER_PARAM / tokens_per_param))
    warning: str | None = None
    selected_tokens_per_param = tokens_per_param * selected_epochs
    if selected_epochs < epochs_to_reference:
        warning = (
            f"Selected {selected_epochs} epochs gives {selected_tokens_per_param:.2f} tokens/parameter, "
            f"below the ~{CHINCHILLA_TOKENS_PER_PARAM:.1f}x reference."
        )

    return ComputeBudgetPlan(
        dataset_tokens=dataset_tokens,
        param_count=param_count,
        tokens_per_param_per_epoch=tokens_per_param,
        reference_tokens_per_param=CHINCHILLA_TOKENS_PER_PARAM,
        epochs_to_reference=epochs_to_reference,
        selected_epochs=selected_epochs,
        selected_tokens_per_param=selected_tokens_per_param,
        warning=warning,
    )


def format_budget_plan(plan: ComputeBudgetPlan) -> str:
    lines = [
        "Compute budget plan:",
        f"  dataset tokens/events per epoch: {plan.dataset_tokens}",
        f"  trainable parameters: {plan.param_count}",
        (
            "  tokens per parameter per epoch: "
            f"{plan.tokens_per_param_per_epoch:.3f}x vs ~{plan.reference_tokens_per_param:.1f}x reference"
        ),
        f"  epochs to reach/exceed reference: {plan.epochs_to_reference}",
        f"  selected epochs: {plan.selected_epochs}",
        f"  selected total tokens per parameter: {plan.selected_tokens_per_param:.3f}x",
    ]
    if plan.warning:
        lines.append(f"  warning: {plan.warning}")
    return "\n".join(lines)
