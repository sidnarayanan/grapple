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
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from copy import deepcopy
from typing import * 
import math

from transformers.modeling_bert import ACT2FN, BertEmbeddings, BertSelfAttention, prune_linear_layer, gelu_new

from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from longformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv


from .met_layer import METLayer
from ._longformer_helpers import * 


VERBOSE = False




class OskarAttention(LongformerSelfAttention):
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

        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.config = config

        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_layer,
        attention_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        if self.attention_band is not None:
            return super().forward(
                    input_layer, 
                    attention_mask,
                    is_index_masked,
                    is_index_global_attn,
                    is_global_attn
                )

        elif False:
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
                d_mask = sliding_chunks_matmul_qk(ones, float_mask, attn_band, padding_value=0)
                attention_scores += d_mask

            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
            attention_probs = self.dropout(attention_probs)
            
            value_layer = value_layer.float().contiguous()
            context_layer = sliding_chunks_matmul_pv(attention_probs, value_layer, attn_band)

        else:
            mixed_query_layer = self.query(input_layer)
            mixed_key_layer = self.key(input_layer)
            mixed_value_layer = self.value(input_layer)

            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            query_layer = (torch.nn.functional.elu(query_layer) + 1)
            key_layer = (torch.nn.functional.elu(key_layer) + 1)
            key_layer = attention_mask * key_layer

            D_inv = 1. / torch.einsum('...nd,...d->...n', query_layer, key_layer.sum(dim=2))
            context = torch.einsum('...nd,...ne->...de', key_layer, value_layer)
            context_layer = torch.einsum('...de,...nd,...n->...ne', context, query_layer, D_inv)

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
        layernormed_context_layer = self.LayerNorm(input_layer + projected_context_layer_dropout)
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

    def forward(self, hidden_states, attention_mask=None, 
                is_index_masked=None, is_index_global_attn=None,
                is_global_attn=None):
        attention_output = self.attention(
                hidden_states, attention_mask, 
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
            )

        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])


        return (hidden_states, ) + attention_output[2:]  # add attentions if we output them


class OskarLayerGroup(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = nn.ModuleList([OskarLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, 
                is_index_masked=None, is_index_global_attn=None,
                is_global_attn=None):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(
                    hidden_states, 
                    attention_mask, 
                    is_index_masked=is_index_masked,
                    is_index_global_attn=is_index_global_attn,
                    is_global_attn=is_global_attn,
                )
            hidden_states = layer_output[0]

        outputs = (hidden_states, )
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

        attention_mask = attention_mask.type(hidden_states.dtype)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
            )
            hidden_states = layer_group_output[0]

        outputs = (hidden_states, )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class Oskar(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.relu = gelu_new #nn.ReLU() 
        self.tanh = nn.Tanh()

        config.output_attentions = False
        config.output_hidden_states = False
        config.layer_norm_eps = 1e-12
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        config.hidden_act = "gelu_new"

        self.embedder = nn.Linear(config.feature_size, config.embedding_size)
        self.encoder = OskarTransformer(config)
        self.decoders = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size), nn.Linear(config.hidden_size, 1)])

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
        h = self.encoder(h, None, mask, head_mask)[0]
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

        self.config = config 

        self.relu = gelu_new #nn.ReLU() 
        self.tanh = nn.Tanh()

        config.output_attentions = False
        config.output_hidden_states = False
        # config.num_hidden_groups = 1
        # config.inner_group_num = 1
        config.layer_norm_eps = 1e-12
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        config.hidden_act = "gelu_new"

        self.input_bn = nn.BatchNorm1d(config.feature_size) 

        self.embedder = nn.Linear(config.feature_size, config.embedding_size)
        self.embed_bn = nn.BatchNorm1d(config.embedding_size) 

        self.encoder = OskarTransformer(config)
        self.decoders = nn.ModuleList([
                                       nn.Linear(config.hidden_size, config.label_size)
                                       ])
        self.decoder_bn = nn.ModuleList([nn.BatchNorm1d(config.hidden_size) for _ in self.decoders[:-1]])

        if self.config.num_global_objects is not None: 
            self.global_x = torch.FloatTensor(
                    np.random.normal(size=(1, self.config.num_global_objects, self.config.feature_size))
                )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_band = (
            self.config.attention_band
            if isinstance(self.config.attention_band, int)
            else max(self.config.attention_band)
        )

        assert attention_band % 2 == 0, f"`attention_band` should be an even value. Given {attention_band}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_band - seq_len % attention_band) % attention_band
        if padding_len > 0:
            logger.info(
                "Input ids are automatically padded from {} to {} to be a multiple of `config.attention_band`: {}".format(
                    seq_len, seq_len + padding_len, attention_band
                )
            )
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, 0, padding_len), value=0)
            attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens

        return padding_len, input_ids, attention_mask

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor) -> torch.Tensor:
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int], device, dtype) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if False and self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def forward(self, x, attn_mask=None):
        if attn_mask is None:
            attn_mask = torch.ones(x.size()[:-1], device=self.config.device)

        if self.config.num_global_objects is not None: 
            batch_size = x.shape[0]
            global_x = self.global_x.repeat((batch_size, 1, 1)).to(self.config.device)
            global_attn_mask = torch.cat(
                    [torch.ones(global_x.shape[:-1], device=self.config.device), torch.zeros(x.shape[:-1], device=self.config.device)],
                    dim=1
                ).long()
            attn_mask = torch.cat(
                    [torch.ones(global_x.shape[:-1], device=self.config.device).long(), attn_mask],
                    dim=1
                )
            x = torch.cat([global_x, x], dim=1)

            attn_mask = self._merge_to_attention_mask(attn_mask, global_attn_mask)

        # if len(attn_mask.shape) == 3:
        #     attn_mask = attn_mask.unsqueeze(1) # [B, P, P] -> [B, 1, P, P]
        # else:
        #     attn_mask = attn_mask.unsqueeze(1).unsqueeze(-1) # [B, P] -> [B, 1, P, 1]

        # if self.config.attention_band is not None:
        #     # attn_mask = self.get_extended_attention_mask(attn_mask)
        #     attn_mask = (1 - attn_mask) * -1e9 # needed for (sparse) softmax attention
        attn_mask: torch.Tensor = self.get_extended_attention_mask(attn_mask, x.shape[:-1], self.config.device, x.dtype)[
            :, 0, 0, :
        ]

        padding_len, x, attn_mask = self._pad_to_window_size(
            input_ids=x,
            attention_mask=attn_mask,
        )

        head_mask = [None] * self.config.num_hidden_layers

        x = self.input_bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        h = self.embedder(x) 
        h = torch.relu(h)
        h = self.embed_bn(h.permute(0, 2, 1)).permute(0, 2, 1)

        h, = self.encoder(h, attn_mask, head_mask)
        h = self.decoders[0](h)

        if self.config.num_global_objects is not None: 
            h = h[:, self.config.num_global_objects:] 

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
