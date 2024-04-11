"""Measure peak memory on of the forward pass."""

# pip install transformers peft

import torch
from peft import LoraConfig, get_peft_model
from torch import manual_seed
from torch.nn import Module
from transformers import (
    AutoModelForCausalLM,
)

from memsave_torch.nn import convert_to_memory_saving


def print_trainable_parameters(model: Module):
    """Function that prints how many parameters are trainable in the given model

    Args:
        model (Module): The model
    """
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def main():
    """Runs the LLM experiment after replacing layers"""
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

    if memsave:
        model = convert_to_memory_saving(model)

    lora_model = get_peft_model(model, lora_config)
    # print_trainable_parameters(lora_model)

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
