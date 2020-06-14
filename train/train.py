import datetime
import logging
import os
import argparse
import shutil
import sys
sys.path.append('..')

import torch
import pytorch_lightning as pl
from pytorch_lightning.logging.neptune import NeptuneLogger
from pytorch_lightning.profiler import PassThroughProfiler, AdvancedProfiler
from transformers import AlbertModel, AlbertConfig
from consonant.model.modeling import ConsonantAlbert
from consonant.utils import set_seed

logger = logging.getLogger(__name__)

def make_parser():
    
    parser = argparse.ArgumentParser()
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

def main():

    args = make_parser()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    set_seed(args)

    albert_base_configuration = AlbertConfig(
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        vocab_size = args.vocab_size,
        max_position_embeddings= args.max_position_embeddings,
        output_vocab_size = args.output_vocab_size,
        type_vocab_size = args.type_vocab_size,
    )

    logger.info('Albert config %s', albert_base_configuration)
    logger.info('Training args %s', args)

    # Set seed
    
    if not os.path.exists(os.path.join('../', args.output_dir)):
        os.mkdir(os.path.join('../', args.output_dir))
  
    args.output_dir = os.path.join('../', args.output_dir, args.exp_name)
    if os.path.exists(args.output_dir):
        flag_continue = input(f"Model name [{args.exp_name}] already exists. Do you want to overwrite? (y/n): ")
        if flag_continue.lower() == 'y' or flag_continue.lower() == 'yes':
            shutil.rmtree(args.output_dir)
            os.mkdir(args.output_dir)
        else:
            print("Exit pre-training")
            exit()
    else:
        os.mkdir(args.output_dir)

    model = ConsonantAlbert(args, albert_base_configuration)
    
    args_dic = vars(args)
    args_dic['pwd'] = os.getcwd()

    neptune_api_key = os.environ['NEPTUNE_API_TOKEN']
    neptune_project_name = 'kboseong/consonant'
    neptune_experiment_name = args.exp_name

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
        val_check_interval = args.validation_step,
    )

    trainer = pl.Trainer(profiler=False, **train_params)
    if args.do_train:
        trainer.fit(model)

    return

if __name__ == "__main__":
    
    main()