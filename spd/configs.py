"""Config classes of various types"""

import warnings
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from spd.log import logger
from spd.spd_types import ModelPath, Probability


class TMSTaskConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)  # Changed from "forbid" to "allow"
    task_name: Literal["tms"] = Field(
        default="tms",
        description="Task identifier for TMS",
    )
    feature_probability: Probability = Field(
        ...,
        description="Probability that a given feature is active in generated data",
    )
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = Field(
        default="at_least_zero_active",
        description="Strategy for generating synthetic data for TMS training",
    )


class ResidualMLPTaskConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)  # Changed from "forbid" to "allow"
    task_name: Literal["residual_mlp"] = Field(
        default="residual_mlp",
        description="Identifier for the residual-MLP decomposition task",
    )
    feature_probability: Probability = Field(
        ...,
        description="Probability that a given feature is active in generated data",
    )
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = Field(
        default="at_least_zero_active",
        description="Strategy for generating synthetic data for residual-MLP training",
    )


class LMTaskConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)  # Changed from "forbid" to "allow"
    task_name: Literal["lm"] = Field(
        default="lm",
        description="Identifier for the language-model decomposition task",
    )
    max_seq_len: PositiveInt = Field(
        default=512,
        description="Maximum sequence length to truncate or pad inputs to",
    )
    buffer_size: PositiveInt = Field(
        default=1000,
        description="Buffered sample count for streaming dataset shuffling",
    )
    dataset_name: str = Field(
        default="lennart-finke/SimpleStories",
        description="HuggingFace dataset identifier to use for the LM task",
    )
    column_name: str = Field(
        default="story",
        description="Dataset column that contains the text to train on",
    )
    train_data_split: str = Field(
        default="train",
        description="Name of the dataset split used for training",
    )
    eval_data_split: str = Field(
        default="test",
        description="Name of the dataset split used for evaluation",
    )


class Config(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)  # Changed from "forbid" to "allow"
    # --- WandB
    wandb_project: str | None = Field(
        default=None,
        description="Weights & Biases project name (set to None to disable WandB logging)",
    )
    wandb_run_name: str | None = Field(
        default=None,
        description="Explicit name for the WandB run (None generates an automatic name)",
    )
    wandb_run_name_prefix: str = Field(
        default="",
        description="Prefix prepended to an auto-generated WandB run name",
    )

    # --- General ---
    seed: int = Field(default=0, description="Random seed for reproducibility")
    C: PositiveInt = Field(
        ...,
        description="The number of subcomponents per layer",
    )

    # Add model validator to warn about unknown eval metrics
    @model_validator(mode="after")
    def warn_unknown_fields(self) -> Self:
        """Warn about unknown fields that might be custom eval metrics"""
        if hasattr(self, '__pydantic_extra__') and self.__pydantic_extra__:
            unknown_fields = list(self.__pydantic_extra__.keys())
            if any('eval' in field.lower() or 'metric' in field.lower() for field in unknown_fields):
                warnings.warn(
                    f"Found unknown eval metric fields: {unknown_fields}. "
                    f"These custom metrics from the saved model are not available in the current codebase and will be ignored.",
                    UserWarning
                )
        return self
