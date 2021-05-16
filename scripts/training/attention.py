import math

import torch
from torch import nn
from torch.nn import LayerNorm as BertLayerNorm


class BertAttention(nn.Module):
    def __init__(self, text_hidden_size, molecule_hidden_size,
                 attention_probs_dropout_prob, num_attention_heads=12, ):
        super().__init__()
        if text_hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (text_hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(text_hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(text_hidden_size, self.all_head_size)
        self.key = nn.Linear(molecule_hidden_size, self.all_head_size)
        self.value = nn.Linear(molecule_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, text_hidden_size, molecule_hidden_size, attention_probs_dropout_prob, hidden_dropout_prob,
                 num_attention_heads=12):
        super().__init__()
        self.att = BertAttention(text_hidden_size=text_hidden_size, molecule_hidden_size=molecule_hidden_size,
                                 attention_probs_dropout_prob=attention_probs_dropout_prob,
                                 num_attention_heads=num_attention_heads)
        self.output = BertAttOutput(hidden_size=text_hidden_size, hidden_dropout_prob=hidden_dropout_prob)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """

    def __init__(self, text_modality_dim, chem_modality_dim, size_out):
        super().__init__()
        self.text_modality_dim = text_modality_dim
        self.chem_modality_dim = chem_modality_dim
        # if text_modality_dim == chem_modality_dim:
        #     self.resize_chem = False
        # else:
        #     self.resize_chem = True
        self.size_out = size_out

        # Weights hidden state modality 1
        weights_hidden_text = torch.Tensor(text_modality_dim, size_out)
        self.weights_hidden_text = nn.Parameter(weights_hidden_text, requires_grad=True)
        nn.init.kaiming_uniform_(self.weights_hidden_text, a=math.sqrt(5))

        # Weights hidden state modality 2
        # if self.resize_chem:
        weights_hidden_chem = torch.Tensor(chem_modality_dim, size_out)
        self.weights_hidden_chem = nn.Parameter(weights_hidden_chem, requires_grad=True)
        nn.init.kaiming_uniform_(self.weights_hidden_chem, a=math.sqrt(5))

        # Weight for sigmoid
        weight_sigmoid = torch.Tensor(size_out * 2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        # initialize weights
        # nn.init.uniform_(self.weights_hidden1, )
        # nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))
        nn.init.uniform_(self.weight_sigmoid, )

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, text_features, chem_features):
        text_hidden = self.tanh_f(torch.matmul(text_features, self.weights_hidden_text))  # B x size_out
        # if self.resize_chem:
        chem_hidden = self.tanh_f(torch.matmul(chem_features, self.weights_hidden_chem))  # B x size_out
        # else:
        # chem_hidden = chem_features
        x = torch.cat((text_hidden, chem_hidden), dim=1)  # B x 2 * size_out
        z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid))  # B

        return z.view(z.size()[0], 1) * text_features + (1 - z).view(z.size()[0], 1) * chem_hidden
