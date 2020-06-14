#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '../')

import transformers
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig, get_linear_schedule_with_warmup
from transformers.modeling_bert import ACT2FN
import torch
from optimization import Lamb
import argparse
import os
import easydict
from torch.utils.data import DataLoader, ConcatDataset
import pyxis.torch as pxt
from torch.nn import CrossEntropyLoss
from consonant.model.tokenization import NGRAMTokenizer


# In[2]:


tokenizer = NGRAMTokenizer(3)


# In[3]:


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
        #print(prediction_scores.shape, answer_label.shape)
        #print(prediction_scores.view(-1, self.config.output_vocab_size).shape, answer_label.view(-1).shape)

        if answer_label is not None :
            loss_fct = CrossEntropyLoss()
            consonant_loss = loss_fct(prediction_scores.view(-1, self.config.output_vocab_size), answer_label.view(-1))
            #consonant_loss = loss_fct(prediction_scores, answer_label)
            #print(consonant_loss.shape, consonant_loss.mean())
            total_loss = consonant_loss
            outputs = (total_loss,) + outputs

        return outputs  


# In[4]:


albert_base_configuration = AlbertConfig(
    hidden_size=256,
    embedding_size=64,
    num_attention_heads=4,
    intermediate_size=1024,
    vocab_size = 17579,
    max_position_embeddings= 100,
    output_vocab_size = 589,
    type_vocab_size = 1,
)

model = Consonant(albert_base_configuration)


# In[5]:


state_dic = '../output/baseline_01/ckpt-0012000.bin'
model.load_state_dict(torch.load(state_dic)['model_state_dict'])


# In[6]:


model = model.cuda()


# In[7]:


def val_dataloader(args):
        
    # We should filter out only directory name excluding all the *.tar.gz files
    data_dir = os.path.join(args.pretrain_dataset_dir, 'val') 
    subset_list = [subset_dir for subset_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subset_dir))]
    train_dataset = ConcatDataset([pxt.TorchDataset(os.path.join(data_dir, subset_dir)) for subset_dir in subset_list])

    # Very small dataset for debugging
    # toy_dataset = Subset(train_dataset, range(0, 100)) # -> If you want to make 100sample toy dataset. 

    data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )

    return data_loader


# In[8]:


args = easydict.EasyDict({
    "pretrain_dataset_dir": '../dataset/processed/ratings_3_100',
    "train_batch_size": 128,
    "num_workers": 0,
})


# In[9]:


valloader = val_dataloader(args)


# In[10]:


for batch in valloader:
    input_ids = batch['head_ids'].type(torch.LongTensor).cuda()
    answer_label = batch['midtail_ids'].type(torch.LongTensor).cuda()  
    attention_mask = batch['attention_masks'].type(torch.LongTensor).cuda()  
    
    #print(input_ids.shape, attention_mask.shape,  answer_label.shape)
    output = model(input_ids, attention_mask=attention_mask, token_type_ids=None, answer_label=answer_label)

    break


# In[11]:


predict_label = output[1].argmax(dim=2)


# In[12]:


print('===============')
for i in range(answer_label.shape[0]):
    predict_label[i][answer_label[i]==0]=0
    answer_string = tokenizer.decode_sent(input_ids[i].detach().cpu().numpy(), answer_label[i].detach().cpu().numpy())
    predict_string = tokenizer.decode_sent(input_ids[i].detach().cpu().numpy(), predict_label[i].detach().cpu().numpy())
    #print('===============')
    print('answer string\t: '+ answer_string)
    print('predict string\t:' + predict_string)
    print('===============')
    


# In[ ]:




