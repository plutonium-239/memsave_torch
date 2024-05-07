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
    'bert_encoder': lambda: models.transformer_model_fns['bert']().bert.encoder.layer[0],
    'memsave_bert_encoder': lambda: models.transformer_model_fns['memsave_bert']().bert.encoder.layer[0],
    'bart_encoder': lambda: models.transformer_model_fns['bart']().decoder.layers[0],
    'memsave_bart_encoder': lambda: models.transformer_model_fns['memsave_bart']().decoder.layers[0],
    'gpt2_layer': lambda: models.transformer_model_fns['gpt2']().transformer.h[0],
    'memsave_gpt2_layer': lambda: models.transformer_model_fns['memsave_gpt2']().transformer.h[0],
    't5_decoder': lambda: models.transformer_model_fns['t5']().decoder.block[1],
    'memsave_t5_decoder': lambda: models.transformer_model_fns['memsave_t5']().decoder.block[1],
}

def run_single(model_fn, name, x):
    model = model_fn()

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

    for name,model_fn in to_test.items():
        x = torch.rand(7, *models.transformer_input_shape)

        run_single(model_fn, name, x)
        