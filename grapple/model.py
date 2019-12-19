from transformers.modeling_albert import * 
import torch
from torch import nn 


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([
            AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)
        ])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            # Index of the layer inside the group
            layer_idx = int(i - group_idx * layers_per_group)
            
            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states, attention_mask, 
                (None if head_mask is None 
                    else head_mask[group_idx*layers_per_group:(group_idx+1)*layers_per_group])
            )  
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


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
        self.encoder = Transformer(config)
        self.decoder = nn.Linear(config.hidden_size, config.label_size)

        self.device = config.device

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.size()[:-1], device=self.device)
        mask = mask.unsqueeze(-1)
        head_mask = torch.ones_like(mask, device=self.device)

        h = self.embedder(x) 
        h = torch.tanh(h)
        print(h.shape, mask.shape)
        head_mask = None
        h = self.encoder(h, mask, head_mask)
        print(h)
        h = self.decoder(h) 

        return h
