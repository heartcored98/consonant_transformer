#!/usr/bin/env python
# coding: utf-8

# In[35]:


import transformers
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig, get_linear_schedule_with_warmup
from transformers.modeling_bert import ACT2FN
import torch
from .optimization import Lamb
import pytorch_lightning as pl
import argparse
import os
from torch.utils.data import DataLoader, ConcatDataset
import pyxis.torch as pxt
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
import time


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
            loss_fct = CrossEntropyLoss()
            consonant_loss = loss_fct(prediction_scores.view(-1, self.config.output_vocab_size), answer_label.view(-1))
            total_loss = consonant_loss
            outputs = (total_loss,) + outputs

        return outputs  


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
            time.sleep(1)

        elif self.global_step % self.hparams.save_log_steps == 0:
            steps_per_sec = self.hparams.save_log_steps / (time.time() - self.last_time)
            examples_per_sec = steps_per_sec * self.hparams.train_batch_size
            self.logger.experiment.log_metric('steps_per_sec', self.global_step, steps_per_sec)
            self.logger.experiment.log_metric('examples_per_sec', self.global_step, examples_per_sec)
            self.logger.experiment.log_metric('learning_rate', self.global_step, self.lr_scheduler.get_last_lr()[-1])
            self.last_time = time.time()

        input_ids = batch['head_ids'].type(torch.LongTensor).cuda()
        answer_label = batch['midtail_ids'].type(torch.LongTensor).cuda()  
        attention_mask = batch['attention_masks'].type(torch.LongTensor).cuda()  
        
        #self.model = self.model.cuda()
        output = self.model(input_ids, attention_mask=attention_mask, token_type_ids=None, answer_label=answer_label)
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
                        }, ckpt_dir)
           
            self.logger.log_artifact(ckpt_dir, ckpt_dir)

        return {'loss': output[0]}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['head_ids'].type(torch.LongTensor).cuda()
        answer_label = batch['midtail_ids'].type(torch.LongTensor).cuda()  
        attention_mask = batch['attention_masks'].type(torch.LongTensor).cuda()  

        
        output = self.model(input_ids, attention_mask=attention_mask, token_type_ids=None, answer_label=answer_label)
        logits = output[1]

        #print(logits.shape, answer_label.shape)
        labels_hat = torch.argmax(logits, dim=2)
        
        correct_count = torch.sum(answer_label == labels_hat)

        output = OrderedDict({
            "val_loss": output[0],
            "correct_count": correct_count,
            "batch_size": len(answer_label)
            })
        return output


    def validation_epoch_end(self, outputs):
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        '''tqdm_dict = {
                "val_loss": val_loss,
                "val_acc": val_acc
                }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}'''
        result = {"val_loss": val_loss, "val_acc": val_acc}
        
        self.logger.experiment.log_metric('val_loss', self.global_step, val_loss)
        self.logger.experiment.log_metric('val_acc', self.global_step, val_acc)
        
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
        optimizer = Lamb(optimizer_grouped_parameters, lr=self.hparams.learning_rate, betas=(.9, .999), eps=self.hparams.adam_epsilon, adam=True)
        #optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    '''def get_tqdm_dict(self):
        avg_loss = getattr(self.trainer, "avg_loss", 0.0)
        tqdm_dict = {"loss": "{:.3f}".format(avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict'''

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
            shuffle=False
        )
        return data_loader