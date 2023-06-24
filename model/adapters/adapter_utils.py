"""Implementation of different utility functions for adapter layers."""

import torch
import torch.nn as nn
from transformers.activations import get_activation


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    # nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    # linear = nn.Linear(input_dim, output_dim)
    linear = nn.Linear(input_dim, output_dim, bias=False)
    init_linear_layer(linear, std=std)
    return linear


class DenseGenerator(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""

    def __init__(self, config, input_dim, output_dim):
        super(DenseGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projected_source_embedding_dim = config["hyperx"]["projected_source_embedding_dim"]
        self.weight_generator = nn.Sequential(
            linear_layer(self.projected_source_embedding_dim, input_dim * output_dim))  #up:128 256*8  up:8*256
        self.bias_generator = nn.Sequential(
            linear_layer(self.projected_source_embedding_dim, input_dim))

    def forward(self, input):
        return self.weight_generator(input).view(-1, self.input_dim, self.output_dim), \
               self.bias_generator(input).view(-1)


class LayerNormGenerator(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""

    def __init__(self, config):
        super(LayerNormGenerator, self).__init__()
        self.projected_source_embedding_dim = config["hyperx"]["projected_source_embedding_dim"]
        self.weight_generator = linear_layer(self.projected_source_embedding_dim, config["adapter"]["input_dim"])
        self.bias_generator = linear_layer(self.projected_source_embedding_dim, config["adapter"]["input_dim"])

    def forward(self, input):
        # breakpoint()
        return self.weight_generator(input).unsqueeze(1), self.bias_generator(input).unsqueeze(1)
