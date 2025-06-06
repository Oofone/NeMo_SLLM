# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
import torch

from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.recipes import llama3_8b

NAME = "llama3_8b_16k"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Llama3 8B model configuration with 16k sequence length.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Llama3 8B model with 16k sequence length.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=llama3_8b_16k ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    model_config = llama3_8b.model()
    model_config.config.seq_length = 16384
    return model_config


def trainer(
    num_nodes: int = 2,
    num_gpus_per_node: int = 8,
) -> run.Config:
    """
    Configure the NeMo Lightning Trainer for Llama3 8B model with 16k sequence length.

    This function sets up the distributed training strategy optimized for longer sequences.

    Args:
        num_nodes (int, optional): Number of compute nodes to use. Defaults to 2.
        num_gpus_per_node (int, optional): Number of GPUs per node. Defaults to 8.

    Returns:
        run.Config: Configuration for the NeMo Lightning Trainer.

    Examples:
        CLI usage:
            $ nemo llm pretrain trainer=llama3_8b_16k ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=2, num_gpus_per_node=8)
            >>> print(trainer_config)

    Note:
        This configuration uses increased parallelism to handle the longer sequence length efficiently.
    """
    return llama3_8b.trainer(
        tensor_parallelism=4,
        pipeline_parallelism=2,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=None,
        context_parallelism=2,
        sequence_parallelism=True,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 2,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    """
    Create a pre-training recipe for Llama3 8B model with 16k sequence length.

    This function sets up a complete configuration for pre-training, including
    model, trainer, and data settings optimized for 16k sequence length.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        num_nodes (int, optional): Number of compute nodes to use. Defaults to 2.
        num_gpus_per_node (int, optional): Number of GPUs per node. Defaults to 8.

    Returns:
        run.Partial: Partial configuration for pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory llama3_8b_16k
            $ nemo llm pretrain --factory "llama3_8b_16k(num_nodes=2, name='my_16k_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="llama3_8b_16k_pretrain", num_nodes=2)
            >>> print(recipe)

    Note:
        This recipe is optimized for handling longer sequences (16k) compared to the standard 8k version.
    """
    recipe = llama3_8b.pretrain_recipe(name=name, dir=dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)

    recipe.model = model()
    recipe.trainer = trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
    recipe.data = run.Config(MockDataModule, seq_length=16384, global_batch_size=512, micro_batch_size=1)

    return recipe
