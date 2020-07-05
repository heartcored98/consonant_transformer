# source ~/.venv/horovod0.18.1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6-cuda10.0/bin/activate
export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMjk1YzJhYWEtYTg3Mi00ZWMwLWEzZTEtYmY1OWE4ZDNkYWM5In0=
python train.py --do_train \
 --ngram 1 \
 --gpus 0 \
 --exp_name small_n1 \
 --learning_rate 6e-4 \
 --num_hidden_layers 12 \
 --train_batch_size 1500 \
 --num_attention_heads 4 \
 --hidden_size 256 \
 --intermediate_size 1024 \
 --save_checkpoint_steps 10000  #--vocab_size=456979  --pretrain_dataset_dir=/home/jovyan/dingbro/consonant_transformer/dataset/processed/ratings_4_100 --exp_name=128_1_128_lr_quadgram --gpus=2
