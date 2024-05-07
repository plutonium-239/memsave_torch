# ruff: noqa
import argparse

import torch
from torchview import draw_graph
from torchviz import make_dot

from experiments.util import models
import torchvision.models as tvm
import transformers.models as tfm
from transformers import AutoConfig

to_test = {
    'bert_encoder': ['bert', lambda model: model.bert.encoder.layer[0]],
    'memsave_bert_encoder': ['memsave_bert', lambda model: model.bert.encoder.layer[0]],
    'bart_encoder': ['bart', lambda model: model.decoder.layers[0]],
    'memsave_bart_encoder': ['memsave_bart', lambda model: model.decoder.layers[0]],
    'gpt2_layer': ['gpt2', lambda model: model.transformer.h[0]],
    'memsave_gpt2_layer': ['memsave_gpt2', lambda model: model.transformer.h[0]],
    't5_decoder': ['t5', lambda model: model.decoder.block[1]],
    'memsave_t5_decoder': ['memsave_t5', lambda model: model.decoder.block[1]],
}

def run_single(model, name, x):
    y = model(x)
    dot = make_dot(
        y.mean(),
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )
    dot.render(filename=name, directory="torchviz-output")


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model", type=str, default="deeprelumodel", help="Which model to use"
    # )

    # args = parser.parse_args()

    # models.conv_input_shape = (3, 64, 64)
    models.transformer_input_shape = (5000, 1024)

    for name in to_test:
        model_name, block_fn = to_test[name]
        config = models.get_transformers_config(model_name)

        models.transformer_input_shape = (config.vocab_size, config.hidden_size)
        x = torch.rand(7, *models.transformer_input_shape)

        model = models.transformer_model_fns.get(model_name)
        run_single(block_fn(model()), name, x)
        