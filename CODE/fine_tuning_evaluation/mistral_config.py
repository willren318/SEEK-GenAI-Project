# -*- coding: utf-8 -*-

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from common import MissionType


class MistralConfig:
    def __init__(self, mission_type: MissionType) -> None:
        self.mission_type = mission_type

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        )

    pass
