# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.insert(0, '../')

import transformers
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig
import torch
import argparse
import os
from torch.utils.data import DataLoader, ConcatDataset, Subset
import pyxis.torch as pxt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from consonant.model.tokenization import NGRAMTokenizer
from consonant.model.modeling import Consonant

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# %%
def load_tokenizer_model(ckpt):
    state = torch.load(ckpt, map_location=torch.device('cpu'))
    tokenizer = NGRAMTokenizer(state['ngram'])

    config = AlbertConfig(**state['config_dict'])
    model = Consonant(config)
    model.load_state_dict(state['model_state_dict'])

    step = int(ckpt.split('-')[-1].split('.')[0])

    return tokenizer, model, state['ngram'], step


def get_dataloader(ngram, batch_size, num_workers):        
    dataset_dir = f'../dataset/processed/comments_{ngram}_100'
    data_dir = os.path.join(dataset_dir, 'val') 
    subset_list = [subset_dir for subset_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subset_dir))]
    val_dataset = ConcatDataset([pxt.TorchDataset(os.path.join(data_dir, subset_dir)) for subset_dir in subset_list])
    val_dataset = Subset(val_dataset, indices=list(range(0, len(val_dataset), 10) ))

    data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    return data_loader


# %%
def evaluate(model, val_dataloader):
    list_length = list()
    list_acc = list()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            input_ids = batch['head_ids'].type(torch.LongTensor).cuda()
            answer_label = batch['midtail_ids'].type(torch.LongTensor).cuda()  
            attention_mask = batch['attention_masks'].type(torch.LongTensor).cuda()  
            
            answer_label[answer_label==0]=-100

            #print(input_ids.shape, attention_mask.shape,  answer_label.shape)
            output = model(input_ids, attention_mask=attention_mask, token_type_ids=None, answer_label=answer_label)
            logits = output[1]
            labels_hat = torch.argmax(logits, dim=2)
            correct = torch.sum(answer_label == labels_hat, dim=1).detach().cpu().numpy() 
            length = torch.sum(answer_label != -100, dim=1).detach().cpu().numpy() 

            acc = correct / length
            list_length.append(length)
            list_acc.append(acc)

            
    list_length = np.concatenate(list_length)
    list_acc = np.nan_to_num(np.concatenate(list_acc))
    return list_length, list_acc


# %%
def evaluate_model(model_size, batch_size, num_workers):

    ckpt_dir = f'../artifact/{model_size}'
    ckpts = sorted([ os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir)])

    print("start ", model_size)
    list_metric = list()
    for ckpt in ckpts:
        tokenizer, model, ngram, step = load_tokenizer_model(ckpt)
        val_dataloader = get_dataloader(ngram, batch_size, num_workers)

        model.eval()
        model.cuda()

        list_length, list_acc = evaluate(model, val_dataloader)
        list_metric.append({'acc':list_acc.mean(), 'model':model_size, 'step': step})
        df = pd.DataFrame(list_metric)
        df.to_csv(f'{model_size}.csv', index=False)
        print(step, list_acc.mean())

    return list_metric


# %%

model_size = 'medium'
batch_size = 3000
num_workers = 8
metric = evaluate_model(model_size, batch_size, num_workers)
df = pd.DataFrame(metric)
df.to_csv(f'{model_size}.csv', index=False)


# # %%
# sns.lineplot(x="step", y="acc",
#              hue="model", markers=True,
#              data=df, dashes=False)
# plt.grid()
