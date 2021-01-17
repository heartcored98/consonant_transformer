FROM huggingface/transformers-pytorch-cpu:2.11.0
RUN apt-get update 
RUN apt-get install openjdk-11-jdk -y
COPY . .
RUN python3 setup.py install 
RUN pip install torchserve torch-model-archiver

ADD https://consonant-transformer-model-artifacts.s3.ap-northeast-2.amazonaws.com/extended-base-0590000.bin ./deploy/model_store/model.bin
RUN torch-model-archiver --model-name medium_consonant --version 1.0 --serialized-file ./deploy/model_store/model.bin --extra-files deploy/setup_config.json --handler ./deploy/transformer_handler.py --export-path ./deploy/model_store

EXPOSE 8080 8081
# ENTRYPOINT ["torchserve", "--start", "--model-store", "./deploy/model_store", "--models", "medium_consonant=medium_consonant.mar"]

ENTRYPOINT ["./dockerd-entrypoint.sh"]
CMD ["serve"]