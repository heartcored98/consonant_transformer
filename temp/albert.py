#!/usr/bin/env python
# coding: utf-8

# In[28]:


import transformers
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig
from transformers.modeling_bert import ACT2FN
import torch


# In[24]:


class AlbertConsonantHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
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


# In[26]:


class Consonant(nn.Module):
    def __init__(self, config):
        super(Consonant, self).__init__()
        self.albert = AlbertModel(config)
        self.predictions = AlbertConsonantHead(config) 

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, answer_label=None):
        outputs = self.albert(input_ids, attention_mask, token_type_ids)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(sequence_output)
        
        outputs = (prediction_scores) + outputs[2:]  

        if answer_label is not None :
            loss_fct = CrossEntropyLoss()
            consonant_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), answer_label.view(-1))
            total_loss = consonant_loss
            outputs = (total_loss,) + outputs

        return outputs  


# In[33]:


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

        input_ids = batch['input_ids']
        answer_label = batch['answer_label']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        output = self.model(input_ids, attention_mask, token_type_ids, answer_label)
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
        input_ids = batch['input_ids']
        answer_label = batch['answer_label']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        output = self.model(input_ids, attention_mask, token_type_ids, answer_label)
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

