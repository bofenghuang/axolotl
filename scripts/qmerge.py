#!/usr/bin/env python

# Modified from https://raw.githubusercontent.com/jondurbin/qlora/main/qmerge.py

import copy
import json
import os

import bitsandbytes as bnb
import fire
import peft
import torch
from bitsandbytes.functional import dequantize_4bit
from peft import PeftModel
from peft.utils import _get_submodules
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def dequantize_model(model, tokenizer, to, dtype=torch.bfloat16, device="cuda"):
    """
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    'to': directory to save the dequantized model
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """
    # if os.path.exists(to):
    #     return AutoModelForCausalLM.from_pretrained(to, torch_dtype=torch.bfloat16, device_map="auto")
    # os.makedirs(to, exist_ok=True)
    cls = bnb.nn.Linear4bit
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)
                quant_state.dtype = dtype
                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)
                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)
                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)
        model.is_loaded_in_4bit = False
        # print("Saving dequantized model...")
        # model.save_pretrained(to)
        # tokenizer.save_pretrained(to)
        # config_data = json.loads(open(os.path.join(to, "config.json"), "r").read())
        # config_data.pop("quantization_config", None)
        # config_data.pop("pretraining_tp", None)
        # with open(os.path.join(to, "config.json"), "w") as config:
        #     config.write(json.dumps(config_data, indent=2))
        return model


def main(
    base_model_name_or_path: str,
    adapter_name_or_path: str,
    merged_model_name_or_path: str,
):

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    # need to be identical from training config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    print("Dequantizing base model...")
    # model = dequantize_model(model, tokenizer, to=f"{base_model_name_or_path}-dequantized")
    model = dequantize_model(model, None, None)

    print("Loading adapter...")
    model = PeftModel.from_pretrained(model=model, model_id=adapter_name_or_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print("Saving merged model...")
    model.save_pretrained(merged_model_name_or_path, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(merged_model_name_or_path)
    config_data = json.loads(open(os.path.join(merged_model_name_or_path, "config.json"), "r").read())
    config_data.pop("quantization_config", None)
    config_data.pop("pretraining_tp", None)

    with open(os.path.join(merged_model_name_or_path, "config.json"), "w") as config:
        config.write(json.dumps(config_data, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
