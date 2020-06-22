FROM pytorch/torchserve:latest-cpu
pip install tqdm
pip install transformers


python setup.py install 

COPY . .
RUN torch-model-archiver --model-name medium_consonant --version 1.0 --serialized-file ./deploy/model_store/medium_consonant_0189000.bin --extra-files deploy/setup_config.json --handler ./deploy/transformer_handler.py --export-path ./deploy/model_store

EXPOSE 8080
CMD ["torchserve", "--start", "--model-store", "./deploy/model_store", "--models", "medium_consonant=medium_consonant.mar"]
