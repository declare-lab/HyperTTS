"""Implements an Adapter and Hyper-adapter Layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation
from .adapter_utils import Activations, linear_layer, LayerNormGenerator, DenseGenerator


class MetaAdapterController(nn.Module):
	"""Implements Meta Adapter controller module, in which
	the adapter layers' weights are generated from a hyper-network.
	In this case, task-embeddings are fixed, and the task
	embeddings will be initialized to random."""

	def __init__(self, config):
		super().__init__()
		self.adapters = nn.ModuleDict(dict())
		self.config = config
		self.input_dim = config["adapter"]["input_dim"]
		self.adapter_dim = config["adapter"]["adapter_dim"]
		self.meta_up_sampler = DenseGenerator(config, self.input_dim, self.adapter_dim)
		self.meta_down_sampler = DenseGenerator(config, self.adapter_dim, self.input_dim)
		self.activation_type = config["adapter"]["adapter_non_linearity"].lower()
		self.add_layer_norm_before_adapter = config["adapter"]["add_layer_norm_before_adapter"]
		self.add_layer_norm_after_adapter = config["adapter"]["add_layer_norm_after_adapter"]
		self.conditional_layer_norm = config["adapter"]["conditional_layer_norm"]
		if self.add_layer_norm_after_adapter:
			if self.conditional_layer_norm:
				self.post_layernorm_hypernet = LayerNormGenerator(config)
			else:
				self.post_layer_norm = nn.LayerNorm(self.input_dim)
		if self.add_layer_norm_before_adapter:
			if self.conditional_layer_norm:
				self.pre_layernorm_hypernet = LayerNormGenerator(config)
			else:
				self.pre_layer_norm = nn.LayerNorm(self.input_dim)

	def call_adapter(self, inputs, source_embedding):
		weight_up, bias_up = self.meta_up_sampler(source_embedding)
		weight_down, bias_down = self.meta_down_sampler(source_embedding)
		
		batch_size = inputs.size()[0]
		down = torch.matmul(inputs, weight_down.transpose(1,2))
		down = down+bias_down.view(batch_size,1,-1)
		
		middle = get_activation(self.activation_type)(down)
		
		output = torch.matmul(middle, weight_up.transpose(1,2)) + bias_up.view(batch_size,1,-1)
		return output

	def apply_pre_layer_norm(self, inputs, source_embedding):
		"""Applies pre layer norm to the inputs."""
		if self.conditional_layer_norm:
			weight, bias = self.pre_layernorm_hypernet(source_embedding)

			mean = torch.mean(inputs, dim=[2], keepdim=True)
			var = torch.mean((inputs - mean) ** 2, dim=[2], keepdim=True)+ 1e-5
			out = ((inputs - mean) / torch.sqrt(var)) * weight + bias
			return out
		else:
			return self.pre_layer_norm(inputs)

	def apply_post_layer_norm(self, inputs, source_embedding):
		"""Applies post layer norm to the inputs."""
		if self.conditional_layer_norm:
			weight, bias = self.post_layernorm_hypernet(source_embedding)  # weight:(128,256), our weight:(8, 128, 256)
			
			mean = torch.mean(inputs, dim=[2], keepdim=True)
			var = torch.mean((inputs - mean) ** 2, dim=[2], keepdim=True)+ 1e-5
			out = ((inputs - mean) / torch.sqrt(var)) * weight + bias
			return out
		else:
			return self.post_layer_norm(inputs)

	def forward(self, source_embedding, inputs):
		z = self.apply_pre_layer_norm(inputs, source_embedding) if self.add_layer_norm_before_adapter else inputs
		outputs = self.call_adapter(z, source_embedding)
		if self.add_layer_norm_after_adapter:
			outputs = self.apply_post_layer_norm(outputs, source_embedding)
		outputs = outputs + inputs
		
		return outputs
