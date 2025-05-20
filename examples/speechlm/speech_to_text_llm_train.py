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

"""
This is an example script for training (SFT/PEFT) multi-modal speech-to-text LLM using NeMo.
All SpeechLMs that has the three componnets (audio encoder, modality adapter and LLM) are supported.
Some example models are:
- SALM (https://arxiv.org/abs/2310.09424)
- VoiceTextBlender (https://arxiv.org/abs/2410.17485)

Example usage: conf.sh

"""


from nemo.collections.speechlm.recipes import speech_to_text_llm_train
from nemo.core.config import hydra_runner


@hydra_runner(config_path="./conf/salm", config_name="whisper-large-v3_linear_llama3.1-8b_stage1")
def main(cfg):
    """main function for training."""
    return speech_to_text_llm_train(cfg)


if __name__ == "__main__":
    main()
