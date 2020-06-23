
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import transformers
from transformers import AlbertModel, AlbertConfig
from transformers.modeling_bert import ACT2FN


class AlbertConsonantHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.output_vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.output_vocab_size)
        self.activation = ACT2FN[config.hidden_act]

        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states

        return prediction_scores

class Consonant(nn.Module):
    def __init__(self, config):
        super(Consonant, self).__init__()
        self.config = config
        self.albert = AlbertModel(config)
        self.predictions = AlbertConsonantHead(config) 

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, answer_label=None):
        outputs = self.albert(input_ids, attention_mask, token_type_ids)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(sequence_output)
        
        outputs = (prediction_scores, ) + outputs[2:]  


        if answer_label is not None :
            answer_label[answer_label==0]=-100
            loss_fct = CrossEntropyLoss()
            consonant_loss = loss_fct(prediction_scores.view(-1, self.config.output_vocab_size), answer_label.view(-1))
            total_loss = consonant_loss
            outputs = (total_loss,) + outputs
        return outputs  
