from transformers.modeling_albert import * 
import torch
from torch import nn 

Transformer = AlbertTransformer


class Joe(nn.Module):
    def __init__(self, config):
        super().__init__()

        config.output_attentions = False
        config.output_hidden_states = False
        config.num_hidden_groups = 1
        config.inner_group_num = 1
        config.layer_norm_eps = 1e-12
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        config.hidden_act = "gelu_new"

        self.embedder = nn.Linear(config.feature_size, config.embedding_size)
        self.encoders = nn.ModuleList([Transformer(config) for _ in range(config.num_encoders)])
        self.decoder = nn.Linear(config.hidden_size, config.label_size)

        self.config = config

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.size()[:-1], device=self.config.device)
        mask = mask.unsqueeze(1).unsqueeze(2) # [B, P] -> [B, 1, P, 1]
        mask = (1 - mask) * -1e5

        head_mask = [None] * self.config.num_hidden_layers

        h = self.embedder(x) 
        h = torch.tanh(h)
        for e in self.encoders:
            h = e(h, mask, head_mask)[0]
        h = self.decoder(h) 

        return h
