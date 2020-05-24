# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
# and Siddharth Narayanan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Re-implementation of ALBERT
    with support for sparse attention, 
    encoded by a provided adjacency matrix

    Oskar: sparse attention for graph regression
    Bruno: sparse attention for vertex classification
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_bert import ACT2FN, BertEmbeddings, BertSelfAttention, prune_linear_layer
from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from longformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv


from .met_layer import METLayer

VERBOSE = False


class OskarAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)

        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()
        self.attention_band = config.attention_band

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.attention_band is not None:
            query_layer = query_layer.permute(0, 2, 1, 3)
            key_layer = key_layer.permute(0, 2, 1, 3)
            value_layer = value_layer.permute(0, 2, 1, 3)

            attn_band = self.attention_band 
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
                remove_from_windowed_attention_mask = (attention_mask != 0)
            query_layer /= math.sqrt(self.attention_head_size)
            query_layer = query_layer.float().contiguous() 
            key_layer = key_layer.float().contiguous() 
            if False:
                attention_scores = diagonaled_mm_tvm(
                        query_layer, key_layer,
                        attn_band, 
                        1, False, 0, False # dilation, is_t1_diag, padding, autoregressive
                    )
            else:
                attention_scores = sliding_chunks_matmul_qk(
                        query_layer, key_layer,
                        attn_band, padding_value=0
                )
            mask_invalid_locations(attention_scores, attn_band, 1, False)
            if attention_mask is not None:
                remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
                float_mask = remove_from_windowed_attention_mask.type_as(query_layer).masked_fill(remove_from_windowed_attention_mask, -10000.0)
                float_mask = float_mask.repeat(1, 1, 1, 1) # don't think I need this
                ones = float_mask.new_ones(size=float_mask.size())  
                if False:
                    d_mask = diagonaled_mm_tvm(ones, float_mask, attn_band, 1, False, 0, False)
                else:
                    d_mask = sliding_chunks_matmul_qk(ones, float_mask, attn_band, padding_value=0)
                attention_scores += d_mask

            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
            attention_probs = self.dropout(attention_probs)
            
            value_layer = value_layer.float().contiguous()
            if False:
                context_layer = diagonaled_mm_tvm(attention_probs, value_layer, attn_band, 1, True, 0, False)
            else:
                context_layer = sliding_chunks_matmul_pv(attention_probs, value_layer, attn_band)

        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            if VERBOSE:
                # print(attention_probs[0, :8, :8])
                print(torch.max(attention_probs), torch.min(attention_probs))

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3)

        context_layer = context_layer.contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)


class OskarLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = OskarAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        try:
            self.activation = ACT2FN[config.hidden_act]
        except KeyError:
            self.activation = config.hidden_act

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them


class OskarLayerGroup(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = nn.ModuleList([OskarLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index])
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class OskarTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([OskarLayerGroup(config) for _ in range(config.num_hidden_groups)])

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

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
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


class Oskar(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.relu = nn.ReLU() 
        self.tanh = nn.Tanh()

        config.output_attentions = False
        config.output_hidden_states = False
        config.num_hidden_groups = 1
        config.inner_group_num = 1
        config.layer_norm_eps = 1e-12
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        config.hidden_act = self.tanh #"gelu_new"

        self.embedder = nn.Linear(config.feature_size, config.embedding_size)
        self.encoders = nn.ModuleList([OskarTransformer(config) for _ in range(config.num_encoders)])
        self.decoders = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size), nn.Linear(config.hidden_size, 1)])

        self.tests = nn.ModuleList(
                    [
                      nn.Linear(config.feature_size, 1, bias=False),
                      # nn.Linear(config.feature_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, 1)
                    ]
                    )

        self.loss_fn = nn.MSELoss()

        self.config = config

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, mask=None, y=None):
        # h = x[:,0,:]
        # # h[:,-1] = y
        # for t in self.tests[:-1]:
        #     h = t(h)
        #     h = self.relu(h)
        # h = self.tests[-1](h).squeeze(-1)

        # if y is None:
        #     return h
        # else:
        #     loss = self.loss_fn(h, y)
        #     return loss, h

        if mask is None:
            mask = torch.ones(x.size()[:-1], device=self.config.device)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1) # [B, P, P] -> [B, 1, P, P]
        else:
            mask = mask.unsqueeze(1).unsqueeze(2) # [B, P] -> [B, 1, P, 1]
        mask = (1 - mask) * -1e9

        head_mask = [None] * self.config.num_hidden_layers

        h = self.embedder(x) 
        h = torch.tanh(h)
        for e in self.encoders:
            h = e(h, mask, head_mask)[0]
        h = self.decoders[0](h[:, 0, :])
        h = self.tanh(h)
        h = self.decoders[1](h).squeeze(-1) 


        if y is None:
            return h
        else:
            loss = self.loss_fn(h, y)
            return loss, h


class Bruno(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.relu = nn.ReLU() 
        self.tanh = nn.Tanh()

        config.output_attentions = False
        config.output_hidden_states = False
        config.num_hidden_groups = 1
        config.inner_group_num = 1
        config.layer_norm_eps = 1e-12
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        config.hidden_act = "gelu_new"

        self.embedder = nn.Linear(config.feature_size, config.embedding_size)
        self.encoders = nn.ModuleList([OskarTransformer(config) for _ in range(config.num_encoders)])
        self.decoders = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size), 
                                       nn.Linear(config.hidden_size, config.label_size)])

        self.tests = nn.ModuleList(
                    [
                      nn.Linear(config.feature_size, 1, bias=False),
                      # nn.Linear(config.feature_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, 1)
                    ]
                    )

        self.config = config

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.size()[:-1], device=self.config.device)
        if len(mask.shape) == 3:
            attn_mask = mask.unsqueeze(1) # [B, P, P] -> [B, 1, P, P]
        else:
            attn_mask = mask.unsqueeze(1).unsqueeze(2) # [B, P] -> [B, 1, P, 1]
        attn_mask = (1 - attn_mask) * -1e9

        head_mask = [None] * self.config.num_hidden_layers

        h = self.embedder(x) 
        h = torch.tanh(h)
        for e in self.encoders:
            h = e(h, attn_mask, head_mask)[0]
        h = self.decoders[0](h)
        h = self.relu(h)
        h = self.decoders[1](h)

        return h

class Agnes(Bruno):
    def __init__(self, config):
        super().__init__(config)

        self.met = METLayer(config)

        self.default_grad = {k:k.requires_grad for k in self.parameters()}

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = self.default_grad[param]

    def freeze_met(self):
        for param in self.met.parameters():
            param.requires_grad = False 

    def freeze_pu(self):
        met_params = set(list(self.met.parameters()))
        for param in self.parameters():
            if param in met_params:
                continue 
            param.requires_grad = False

    def forward(self, x, q, y, mask=None, particle_mask=None, return_weights=False):
        if mask is not None and particle_mask is None:
            particle_mask = mask 
        yhat = super().forward(x, mask)
        # yhat = super().forward(x[:, :, [0, 1]], mask)

        score = nn.functional.softmax(yhat, dim=-1)[:, :, 1] # [B,P]
        q_abs = torch.abs(q)
        score = (q_abs * y) + ((1 - q_abs) * score)
        # score[q != 0] = y[q != 0]

        methat, weights = self.met(x, score, particle_mask)
        
        if return_weights:
            return yhat, methat, weights
        else:
            return yhat, methat
