source ~/.venv/horovod0.18.1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6-cuda10.0/bin/activate
export NEPTUNE_API_TOKEN=<NEPTUNE_API_TOKEN>
python train.py --do_train --vocab_size=456979 --pretrain_dataset_dir=/home/jovyan/dingbro/consonant_transformer/dataset/processed/ratings_4_100 --exp_name=128_1_128_lr_quadgram --gpus=2
