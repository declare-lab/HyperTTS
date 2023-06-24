"""Implementation of embedding different information sources (task, language, layer_id) for meta adapter layers."""
import os
import json
import torch
import torch.nn as nn
from .adapter_utils import linear_layer


def init_x_embeddings(Xs, x_embedding_dim):
	x2embeddings = nn.ParameterDict(dict())
	for x in Xs:
		x_embedding = torch.empty(x_embedding_dim)
		nn.init.normal_(x_embedding)
		x2embeddings[x] = nn.Parameter(x_embedding)
	return x2embeddings

class SourceController(nn.Module):
	def __init__(self, config, num_layers):
		super(SourceController, self).__init__()
		self.config = config
		speaker_embedding_dim = config["hyperx"]["speaker_embedding_dim"]
		
		self.spk_emb_projector = nn.Linear(256, speaker_embedding_dim)
		
		layer_ids = [str(i) for i in range(num_layers)]
		layer_id_embedding_dim = config["hyperx"]["layer_id_embedding_dim"]
		source_embedding_dim = config["hyperx"]["source_embedding_dim"]
		projected_source_embedding_dim = config["hyperx"]["projected_source_embedding_dim"]
		

		if config["hyperx"]["condition_to_layer_id"]:
			self.layer_id_embeddings = init_x_embeddings(layer_ids, layer_id_embedding_dim)
		if config["hyperx"]["project_source_embeddings"]:
			self.source_embedding_MLP = nn.Sequential(
				linear_layer(source_embedding_dim, projected_source_embedding_dim),
				nn.ReLU(),
				# linear_layer(source_hidden_dim, projected_source_embedding_dim)
				)

	def forward(self, speaker_embeds, layer_id=None):
		## changed
		batch_size = len(speaker_embeds)
		source_emb = self.spk_emb_projector(speaker_embeds)

		if self.config["hyperx"]["condition_to_layer_id"]:
			
			batch_layer_emb = self.layer_id_embeddings[layer_id].repeat(batch_size, 1)
			source_emb = torch.cat([source_emb, batch_layer_emb], dim=1)
			
		if self.config["hyperx"]["project_source_embeddings"]:
			source_emb = self.source_embedding_MLP(source_emb)
		return source_emb
