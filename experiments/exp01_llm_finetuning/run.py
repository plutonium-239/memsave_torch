"""Measure peak memory on of the forward pass."""

# pip install transformers peft

import sys
from os import path

import torch
from peft import LoraConfig, get_peft_model
from torch import manual_seed
from torch.nn import Conv1d, LayerNorm, Linear
from transformers import (
    AutoModelForCausalLM,
)

HEREDIR = path.dirname(path.abspath(__file__))
LIBDIR = path.join(HEREDIR, "memsave_torch")
if LIBDIR not in sys.path:
    sys.path.append(LIBDIR)

from memsave_torch.nn import (
    MemSaveConv1d,
    MemSaveLayerNorm,
    MemSaveLinear,
    recursive_setattr,
)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def main():
    manual_seed(0)

    memsave = True

    # config = GPT2Config.from_pretrained("gpt2")
    # config.hidden_dropout_prob = 0
    # config.attention_probs_dropout_prob = 0
    # model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        # target_modules=["c_attn"],  # LoRA on the attention weights, GPT2
        target_modules=["q_proj", "v_proj"],  # LoRA on the attention weight, GPT neo
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(model, lora_config)
    # print_trainable_parameters(lora_model)

    if memsave:
        for name, layer in model.named_modules():
            if isinstance(layer, Linear):
                new_layer = MemSaveLinear.from_nn_Linear(layer)
                for p1, p2 in zip(layer.parameters(), new_layer.parameters()):
                    p2.requires_grad = p1.requires_grad
                recursive_setattr(model, name, new_layer)
            elif isinstance(layer, Conv1d):
                new_layer = MemSaveConv1d.from_nn_Conv1d(layer)
                for p1, p2 in zip(layer.parameters(), new_layer.parameters()):
                    p2.requires_grad = p1.requires_grad
                recursive_setattr(model, name, new_layer)
            elif isinstance(layer, LayerNorm):
                new_layer = MemSaveLayerNorm.from_nn_LayerNorm(layer)
                for p1, p2 in zip(layer.parameters(), new_layer.parameters()):
                    p2.requires_grad = p1.requires_grad
                recursive_setattr(model, name, new_layer)

    batch_size = 8
    seq_len = 512
    input_ids = torch.randint(10, (batch_size, seq_len))

    out = lora_model(input_ids)

    # print(out)
    print({type(layer) for layer in model.modules()})

    # for name, layer in model.named_modules():
    #     print(name, type(layer))

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}")

    # print(f"{name} requires_grad = {param.requires_grad}")
    # print(out["logits"].flatten()[0:10])
    return out


if __name__ == "__main__":
    main()
    # max_usage = memory_usage(main, interval=1e-3, max_usage=True)
    # print(f"Peak mem: {max_usage}.")
