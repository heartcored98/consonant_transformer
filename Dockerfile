FROM huggingface/transformers-pytorch-cpu:latest
RUN apt-get update 
RUN apt-get install openjdk-11-jdk -y
COPY . .
RUN python3 setup.py install 
RUN pip install torchserve torch-model-archiver

COPY . .
RUN torch-model-archiver --model-name medium_consonant --version 1.0 --serialized-file ./deploy/model_store/medium_consonant_0189000.bin --extra-files deploy/setup_config.json --handler ./deploy/transformer_handler.py --export-path ./deploy/model_store

EXPOSE 8080 8081
# ENTRYPOINT ["torchserve", "--start", "--model-store", "./deploy/model_store", "--models", "medium_consonant=medium_consonant.mar"]

ENTRYPOINT ["./dockerd-entrypoint.sh"]
CMD ["serve"]