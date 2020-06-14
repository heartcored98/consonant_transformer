#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[12]:


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


# In[13]:


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


# In[14]:


model = model.cuda()


# In[5]:


def make_parser():
    
    parser = argparse.ArgumentParser("")
    #config setting
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=64, type=int)
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--intermediate_size', default=1024, type=int)
    parser.add_argument('--vocab_size', default=17579, type=int)
    parser.add_argument('--max_position_embeddings', default=100, type=int)
    parser.add_argument('--output_vocab_size', default=589, type=int)
    parser.add_argument('--type_vocab_size', default=1, type=int)
    
    #exp setting
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--adam_epsilon', default=1e-6, type=float)
    parser.add_argument('--warmup_steps', default=10, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--max_steps', default=200, type=int)
    parser.add_argument('--save_checkpoint_steps', default=100, type=int)
    parser.add_argument('--validation_step', default=50, type=int)
    parser.add_argument('--save_log_steps', default=1, type=int)

    parser.add_argument('--pretrain_dataset_dir', default='../dataset/processed/ratings_3_100', type=str)
    parser.add_argument('--dataset_type', default='owt', type=str)
    parser.add_argument('--exp_name', default='baseline', type=str)

    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    parser.add_argument('--seed', default=42, type=int, help='random seed for initialization')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')

    parser.add_argument('--num_workers', default=8, type=int)

    args = parser.parse_args()

    return args


# In[6]:


def train_dataloader(args):
        
    # We should filter out only directory name excluding all the *.tar.gz files
    data_dir = os.path.join(args.pretrain_dataset_dir, 'train') 
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

    scheduler = get_linear_schedule_with_warmup(
        args.opt, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )
    args.lr_scheduler = scheduler
    return data_loader


# In[7]:


args = easydict.EasyDict({
    "pretrain_dataset_dir": '../dataset/processed/ratings_3_100',
    "train_batch_size": 128,
    "num_workers": 0,
    "warmup_steps": 100,
    "max_steps": 1000,
    "adam_epsilon":1e-6,
    "weight_decay":1e-8,
    "learning_rate":1e-4
})


# In[8]:


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, betas=(.9, .999), eps=args.adam_epsilon, adam=True)


# In[9]:


args['opt'] = optimizer


# In[10]:


trainloader = train_dataloader(args)


# In[15]:


for batch in trainloader:
    input_ids = batch['head_ids'].type(torch.LongTensor).cuda()
    answer_label = batch['midtail_ids'].type(torch.LongTensor).cuda()  
    attention_mask = batch['attention_masks'].type(torch.LongTensor).cuda()  
    
    print(input_ids.shape, attention_mask.shape,  answer_label.shape)
    output = model(input_ids, attention_mask=attention_mask, token_type_ids=None, answer_label=answer_label)
    output[0].backward()
    optimizer
    break


# In[27]:


output[1].argmax(dim=2).shape


# In[40]:


(torch.sum(answer_label==output[1].argmax(dim=2)).item() / torch.sum(answer_label!=-100).item())


# In[38]:


torch.sum(answer_label==output[1].argmax(dim=2))


# In[39]:


torch.sum(answer_label!=-100)


# # pl

# In[ ]:


class ConsonantAlbert(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace, config):
        super().__init__()

        self.hparams = hparams
        self.config = config

        #self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        self.model = Consonant(config)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        # Log steps_per_sec every 100 steps
        if self.global_step == 0:
            self.last_time = time.time()

        elif self.global_step % self.hparams.save_log_steps == 0:
            steps_per_sec = self.hparams.save_log_steps / int(time.time() - self.last_time)
            examples_per_sec = steps_per_sec * self.hparams.train_batch_size
            self.logger.experiment.log_metric('steps_per_sec', self.global_step, steps_per_sec)
            self.logger.experiment.log_metric('examples_per_sec', self.global_step, examples_per_sec)
            self.logger.experiment.log_metric('learning_rate', self.global_step, self.lr_scheduler.get_last_lr()[-1])
            self.last_time = time.time()

        input_ids = batch['head_ids']
        answer_label = batch['midtail_ids']
        attention_mask = batch['attention_masks']

        output = self.model(input_ids, attention_mask, answer_label)
        # output = self.model(input_ids, attention_mask, token_type_ids, masked_lm_labels)

        self.logger.experiment.log_metric('Total_loss', self.global_step, output[0].item())

        # Save model and optimizer
        if self.global_step % self.hparams.save_checkpoint_steps == 0 and self.global_step != 0:
            
            ckpt = f'ckpt-{self.global_step:07}.bin'
            ckpt_dir = os.path.join(self.hparams.output_dir, ckpt)
            
            torch.save( {'model_state_dict': self.model.state_dict(), 
                         'optimizer_state_dict': self.opt.state_dict(),
                         'scheduler_state_dict' : self.lr_scheduler.state_dict(),
                         'loss': output[0].item()
                        }, output_model_file)
           
            self.logger.log_artifact(ckpt_dir, ckpt_dir)
        
        return {'loss': output[0]}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['head_ids']
        answer_label = batch['midtail_ids']
        attention_mask = batch['attention_masks']

        output = self.model(input_ids, attention_mask, answer_label)
        logits = output[1]
        labels_hat = torch.argmax(logits, dim=1)
        
        f1, exact = comput_score(answer_label, labels_hat)

        output = OrderedDict({
            "val_loss": output[0].item(),
            "f1": f1,
            "exact" : exact
            "batch_size": len(answer_label)
            })
        return output


    def validation_end(self, outputs):
        val_f1 = sum([out["f1"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_exact = sum([out["f1"] for out in outputs]).float() / sum(out["exact"] for out in outputs)
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "val_loss": val_loss,
                "val_f1": val_acc,
                "val_exact": val_acc
                }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}
        
        self.logger.experiment.log_metric('val_loss', self.global_step, val_loss)
        self.logger.experiment.log_metric('val_f1', self.global_step, val_f1)
        self.logger.experiment.log_metric('val_exact', self.global_step, val_exact)
        
        return result

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Lamb(optimizer_grouped_parameters, lr=args.hparams.learning_rate, betas=(.9, .999), eps=self.hparams.adam_epsilonm, adam=True)
        #optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        avg_loss = getattr(self.trainer, "avg_loss", 0.0)
        tqdm_dict = {"loss": "{:.3f}".format(avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        
        # We should filter out only directory name excluding all the *.tar.gz files
        data_dir = os.path.join(self.hparams.pretrain_dataset_dir, 'train') 
        subset_list = [subset_dir for subset_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subset_dir))]
        train_dataset = ConcatDataset([pxt.TorchDataset(os.path.join(data_dir, subset_dir)) for subset_dir in subset_list])

        # Very small dataset for debugging
        # toy_dataset = Subset(train_dataset, range(0, 100)) # -> If you want to make 100sample toy dataset. 

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True
        )

        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.max_steps
        )
        self.lr_scheduler = scheduler
        return data_loader

    def val_dataloader(self):
        
        # We should filter out only directory name excluding all the *.tar.gz files
        data_dir = os.path.join(self.hparams.pretrain_dataset_dir, 'val') 
        subset_list = [subset_dir for subset_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subset_dir))]
        train_dataset = ConcatDataset([pxt.TorchDataset(os.path.join(data_dir, subset_dir)) for subset_dir in subset_list])

        # Very small dataset for debugging
        # toy_dataset = Subset(train_dataset, range(0, 100)) # -> If you want to make 100sample toy dataset. 

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True
        )
        return data_loader


# In[ ]:


# coding=utf-8

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import datetime
import logging
import os
import argparse
import shutil
import sys
sys.path.append('../')


import torch
import pytorch_lightning as pl
from pytorch_lightning.logging.neptune import NeptuneLogger
from pytorch_lightning.profiler import PassThroughProfiler, AdvancedProfiler
from transformers import (
    CONFIG_MAPPING,
    set_seed,
    ElectraConfig,
)


from model import BaseElectra


logger = logging.getLogger(__name__)


def revise_config(config: ElectraConfig, args: argparse.Namespace):
    """
    Revise config as we want
        1. Add multiplier between generator and discriminator
        2. Degree of weight sharing
            'no' : Share nothing
            'embedding' : Share only embedding layer
            'all' : Share all layers
        3. Set configuration as electra-small
    """

    config.multiplier_generator_and_discriminator = args.multiplier_generator_and_discriminator
    config.weight_sharing_degree = args.weight_sharing_degree
    config.rtd_loss_weight = args.rtd_loss_weight
    config.generator_num_hidden_layers = args.generator_num_hidden_layers
    config.save_log_steps = args.save_log_steps

    return config


def make_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mlm', default=True, action='store_true')
    parser.add_argument('--mlm_probability', default=0.15, type=float)

    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--adam_epsilon', default=1e-6, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--save_checkpoint_steps', default=25000, type=int)
    parser.add_argument('--save_log_steps', default=100, type=int)

    parser.add_argument('--pretrain_dataset_dir', default='../dataset/', type=str)
    parser.add_argument('--dataset_type', default='owt', type=str)
<<<<<<< HEAD
    parser.add_argument("--model-name", default='electra_small_owt',
=======
    parser.add_argument("--model-name", default='electra_small_owt_reduce_log_metric',
>>>>>>> 3ab8eedf19bbc47a4adc9d5fd69671ee451d0959
                        help="The name of the model being fine-tuned.")

    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    parser.add_argument('--seed', default=42, type=int, help='random seed for initialization')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--generator_num_hidden_layers', default=12, type=int)
    parser.add_argument('--discriminator_num_hidden_layers', default=12, type=int)

    parser.add_argument('--multiplier_generator_and_discriminator', default=4, type=int)
    parser.add_argument('--weight_sharing_degree', default='embedding', type=str)
    parser.add_argument('--rtd_loss_weight', default=50.0, type=float)

    args = parser.parse_args()

    return args

def main():

    args = make_parser()

    config = CONFIG_MAPPING['electra']()
    config = revise_config(config, args)

    logger.info('Electra config %s', config)
    logger.info('Training args %s', args)

    # Set seed
    set_seed(args.seed)
    if not os.path.exists(os.path.join('../', args.output_dir)):
        os.mkdir(os.path.join('../', args.output_dir))

    # Initialize model directory
    # model_dir = "{}_{}".format(args.model_name, datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S"))    
    args.output_dir = os.path.join('../', args.output_dir, args.model_name)
    if os.path.exists(args.output_dir):
        flag_continue = input(f"Model name [{args.model_name}] already exists. Do you want to overwrite? (y/n): ")
        if flag_continue.lower() == 'y' or flag_continue.lower() == 'yes':
            shutil.rmtree(args.output_dir)
            os.mkdir(args.output_dir)
        else:
            print("Exit pre-training")
            exit()
    else:
        os.mkdir(args.output_dir)

    model = BaseElectra(args, config)
    
    neptune_api_key = os.environ['NEPTUNE_API_TOKEN']
    neptune_project_name = 'IRNLP/electra'
    neptune_experiment_name = 'electra_pytorch'

    neptune_logger = NeptuneLogger(
        api_key=neptune_api_key,
        project_name=neptune_project_name,
        experiment_name=neptune_experiment_name,
        tags=["torch", "pretrain"],
    )

    train_params = dict(
        gpus=args.n_gpu,
        gradient_clip_val=args.max_grad_norm,
        logger=neptune_logger,
        early_stop_callback=None,
    )

    trainer = pl.Trainer(profiler=False, **train_params)
    if args.do_train:
        trainer.fit(model)

    return

if __name__ == "__main__":
    main()


# In[36]:


get_ipython().system('which python')


# In[ ]:


import os
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import argparse

import pyxis.torch as pxt

from transformers import (
    AdamW,
    ElectraPreTrainedModel,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    ElectraTokenizer,
    get_linear_schedule_with_warmup,
)

import sys
sys.path.append('../')

from data_collator import DataCollatorForLanguageModeling

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset, Subset
from torch.utils.data.sampler import RandomSampler

BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ElectraEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.embedding_size, eps=config.layer_norm_eps)


class Electra(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.discriminator_config = copy.deepcopy(config)
        self.generator_config = copy.deepcopy(config)

        self.generator_config.num_hidden_layers = self.config.generator_num_hidden_layers

        self.generator_config.hidden_size //= self.config.multiplier_generator_and_discriminator
        self.generator_config.intermediate_size //= self.config.multiplier_generator_and_discriminator
        self.generator_config.num_attention_heads //= self.config.multiplier_generator_and_discriminator
        
        self.generator = ElectraForMaskedLM(self.generator_config)
        self.discriminator = ElectraForPreTraining(self.discriminator_config)

        if self.config.weight_sharing_degree == 'embedding':
            self.embedding = ElectraEmbeddings(self.config)
            self.generator.electra.embeddings = self.embedding
            self.discriminator.electra.embeddings = self.embedding
        
        elif self.config.weight_sharing_degree == 'all':
            raise NotImplementedError('In case of sharing all weights')

        self.init_weights()

        print(self.generator)
        print(self.discriminator)
        
        print("======== Generator Config ========")
        print(self.generator_config)

        print("======== Discriminator Config ========")
        print(self.discriminator_config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        masked_lm_labels=None,
        logger=None,
        global_step=None,
    ):

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Generator forward
        MLM_loss, MLM_prediction_scores = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            masked_lm_labels=masked_lm_labels
        )
        logger.experiment.log_metric('MLM_loss', MLM_loss.item())

        ## Prepare discriminator input and RTD labels, adding uniform noise to generator's logits
        uniform_noise = torch.rand(MLM_prediction_scores.shape, device=self.device)
        gumbel_noise = -(-(uniform_noise + 1e-9).log() + 1e-9).log()

        sampled_prediction_scores = MLM_prediction_scores + gumbel_noise
        sampled_argmax_words = F.softmax(sampled_prediction_scores, dim=-1).argmax(dim=-1)
        sampled_input_ids_to_discriminator = sampled_argmax_words * (masked_lm_labels != -100) + input_ids * (masked_lm_labels == -100)
        sampled_RTD_labels = (sampled_input_ids_to_discriminator != masked_lm_labels) * (masked_lm_labels != -100)

        # Discriminator Forward
        RTD_loss, RTD_prediction_scores = self.discriminator(
            input_ids=sampled_input_ids_to_discriminator,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=sampled_RTD_labels,
        )
        logger.experiment.log_metric('RTD_loss', global_step, RTD_loss.item())


        if global_step % self.config.save_log_steps == 0:
            # Evaluate Generator's MLM performance based on initial prediction scores
            argmax_words = F.softmax(MLM_prediction_scores, dim=-1).argmax(dim=-1)
            input_ids_to_discriminator = argmax_words * (masked_lm_labels != -100) + input_ids * (masked_lm_labels == -100) # -> masked 된 토큰들만 generator로 바뀐 것.
            RTD_labels = (input_ids_to_discriminator != masked_lm_labels) * (masked_lm_labels != -100)

            incorrect_masked_word = RTD_labels.sum().item() # Since True means incorrect word in RTD_labels
            total_masked_word = (masked_lm_labels != -100).sum().item()
            sampled_incorrect_masked_word = sampled_RTD_labels.sum().item()
            logger.experiment.log_metric('MLM_accuracy', global_step, (total_masked_word - incorrect_masked_word) / total_masked_word * 100)
            logger.experiment.log_metric('Sampled_MLM_accuracy', global_step, (total_masked_word - sampled_incorrect_masked_word) / total_masked_word * 100)

            # Calculate classification metric
            RTD_preds = RTD_prediction_scores.sigmoid() >= 0.5
            correct_true = ((RTD_preds == RTD_labels) * (RTD_preds == 1)).sum().item()
            predicted_true = RTD_preds.sum().item()
            target_true = RTD_labels.sum().item()

            recall = correct_true / target_true if target_true > 0 else 0
            precision = correct_true / predicted_true if predicted_true > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            logger.experiment.log_metric('RTD_accuracy', global_step, (RTD_preds == RTD_labels).sum().item() / (batch_size * seq_len) * 100)
            logger.experiment.log_metric('RTD_precision', global_step, precision * 100)
            logger.experiment.log_metric('RTD_recall', global_step, recall * 100)
            logger.experiment.log_metric('RTD_f1', global_step, f1_score * 100)

        loss = MLM_loss + self.config.rtd_loss_weight * RTD_loss
        output = (loss, RTD_prediction_scores)
        return output

    
class BaseElectra(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace, config):
        super().__init__()

        self.hparams = hparams
        self.config = config

        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        self.model = Electra(config)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        # Log steps_per_sec every 100 steps
        if self.global_step == 0:
            self.last_time = time.time()

        elif self.global_step % self.hparams.save_log_steps == 0:
            steps_per_sec = self.hparams.save_log_steps / int(time.time() - self.last_time)
            examples_per_sec = steps_per_sec * self.hparams.train_batch_size
            self.logger.experiment.log_metric('steps_per_sec', self.global_step, steps_per_sec)
            self.logger.experiment.log_metric('examples_per_sec', self.global_step, examples_per_sec)
            self.logger.experiment.log_metric('learning_rate', self.global_step, self.lr_scheduler.get_last_lr()[-1])
            self.last_time = time.time()

        input_ids = batch['input_ids']
        masked_lm_labels = batch['masked_lm_labels']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        output = self.model(input_ids, attention_mask, token_type_ids, masked_lm_labels, self.logger, self.global_step)
        # output = self.model(input_ids, attention_mask, token_type_ids, masked_lm_labels)

        self.logger.experiment.log_metric('Total_loss', output[0].item())

        # Save model and optimizer
        if self.global_step % self.hparams.save_checkpoint_steps == 0 and self.global_step != 0:

            ckpt = f'ckpt-{self.global_step:07}'
            ckpt_dir = os.path.join(self.hparams.output_dir, ckpt)
            generator_dir = os.path.join(ckpt_dir, 'generator')
            discriminator_dir = os.path.join(ckpt_dir, 'discriminator')
            optimizer_dir = os.path.join(ckpt_dir, 'optimizer.pt')
            
            os.mkdir(ckpt_dir)
            os.mkdir(generator_dir)
            os.mkdir(discriminator_dir)

            # save artifact to local disk
            self.model.generator.save_pretrained(generator_dir)
            self.model.discriminator.save_pretrained(discriminator_dir)
            torch.save(self.opt.state_dict(), optimizer_dir)

            # upload artifact to neptune server
            self.logger.log_artifact(os.path.join(generator_dir, 'config.json'), os.path.join(ckpt, 'generator/config.json'))
            self.logger.log_artifact(os.path.join(generator_dir, 'pytorch_model.bin'), os.path.join(ckpt, 'generator/pytorch_model.bin'))
            self.logger.log_artifact(os.path.join(discriminator_dir, 'config.json'), os.path.join(ckpt, 'discriminator/config.json'))
            self.logger.log_artifact(os.path.join(discriminator_dir, 'pytorch_model.bin'), os.path.join(ckpt, 'discriminator/pytorch_model.bin'))
            self.logger.log_artifact(os.path.join(ckpt_dir, 'optimizer.pt'), os.path.join(ckpt, 'optimizer.pt'))
        
        return {'loss': output[0]}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        avg_loss = getattr(self.trainer, "avg_loss", 0.0)
        tqdm_dict = {"loss": "{:.3f}".format(avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_batch_size = self.hparams.train_batch_size
        
        if self.hparams.dataset_type == 'owt':
            data_dir = os.path.join(self.hparams.pretrain_dataset_dir, 'openwebtext_lmdb_128')
        else:
            raise NotImplementedError('Bookcorpus, Wiki dataset is not implemented')

        # We should filter out only directory name excluding all the *.tar.gz files
        subset_list = [subset_dir for subset_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subset_dir))]

        train_dataset = ConcatDataset([pxt.TorchDataset(os.path.join(data_dir, subset_dir)) for subset_dir in subset_list])
        # train_dataset = Subset(train_dataset, range(0, 2))
        # assert len(train_dataset) == 48185029, "length of dataset size is not matched!" # -> This should be 48185029 lines

        # Very small dataset for debugging
        # train_dataset = pxt.TorchDataset(os.path.join('../dataset', 'openwebtext_lmdb_128/1_of_14'))
        # train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, 1024)))

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=self.hparams.mlm, mlm_probability=self.hparams.mlm_probability
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            collate_fn=data_collator.collate_batch,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True
        )

        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.max_steps
        )
        self.lr_scheduler = scheduler
        return data_loader

