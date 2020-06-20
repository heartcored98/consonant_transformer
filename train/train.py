import argparse
import datetime
import os
import shutil
import sys
sys.path.append('..') # in order to make consonant visible 


import torch
from transformers import AlbertModel, AlbertConfig
import pytorch_lightning as pl
from pytorch_lightning.logging.neptune import NeptuneLogger
from pytorch_lightning.profiler import PassThroughProfiler, AdvancedProfiler
from pytorch_lightning import seed_everything


from consonant.model.modeling import ConsonantAlbert
from consonant.model.tokenization import NGRAMTokenizer


def make_parser():    
    parser = argparse.ArgumentParser()

    # model archiecture configuration
    parser.add_argument('--max_position_embeddings', default=100, type=int)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--intermediate_size', default=2048, type=int)
    parser.add_argument('--num_attention_heads', default=8, type=int)
    parser.add_argument('--num_hidden_layers', default=12, type=int)
    parser.add_argument('--num_hidden_groups', default=1, type=int)
    parser.add_argument('--ngram', default=3, type=int) 
    parser.add_argument('--output_vocab_size', default=589, type=int)
    parser.add_argument('--type_vocab_size', default=1, type=int)
    parser.add_argument('--classifier_dropout_prob', default=0.1, type=float)

    # train/validation configuration
    parser.add_argument('--train_batch_size', default=390, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--adam_epsilon', default=1e-6, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--save_checkpoint_steps', default=100, type=int)
    parser.add_argument('--validation_step', default=20000, type=int)
    parser.add_argument('--save_log_steps', default=50, type=int)
    parser.add_argument('--grad_accum_steps', type=int, default=1)

    # experiment configuration
    parser.add_argument('--exp_name', default='comment_baseline_b390_savecheck', type=str)
    parser.add_argument('--pretrain_dataset_dir', default='/home/whwodud98/consonant_transformer/dataset/processed/comments_3_100', type=str)
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int, help='random seed for initialization')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--benchmark', default=False, type=bool)
    args = parser.parse_args()

    args.vocab_size = len(NGRAMTokenizer(args.ngram).head2id) # quad-gram : 456979 / tri-gram : 17579 / bi-gram : 679 / uni-gram : 29 
    return args


def main():

    args = make_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    seed_everything(args.seed)

    # Prepare output directory
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

    # Setup for neptune logger
    neptune_api_key = os.environ['NEPTUNE_API_TOKEN']
    neptune_project_name = 'kevinjo/cs372'
    neptune_experiment_name = args.exp_name
    neptune_logger = NeptuneLogger(
        api_key=neptune_api_key,
        project_name=neptune_project_name,
        experiment_name=neptune_experiment_name,
        tags=["torch", "pretrain"],
        params=vars(args)
    )

    # Setup for pytorch-lightning params
    train_params = dict(
        logger=neptune_logger,
        gpus=args.n_gpu,
        gradient_clip_val=args.max_grad_norm,
        early_stop_callback=None,
        checkpoint_callback=False,
        val_check_interval=args.validation_step,
        accumulate_grad_batches=args.grad_accum_steps,
        max_steps=args.max_steps,
        benchmark=args.benchmark,
    )

    # Setup for albert model 
    albert_base_configuration = AlbertConfig(
        classifier_dropout_prob = args.classifier_dropout_prob,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        num_hidden_groups=args.num_hidden_groups,
        intermediate_size=args.intermediate_size,
        vocab_size = args.vocab_size,
        max_position_embeddings= args.max_position_embeddings,
        output_vocab_size = args.output_vocab_size,
        type_vocab_size = args.type_vocab_size,
    )
    model = ConsonantAlbert(args, albert_base_configuration)

    # Start model training
    trainer = pl.Trainer(profiler=False, **train_params)
    if args.do_train:
        trainer.fit(model)
    return


if __name__ == "__main__":
    main()