v0.2.2  
- Fix huggingface/transfomers-pytorch-cpu version on Dockerbuild (since transformer v4 does not compatible with v2)  
- Add docker build, push, inference instruction  

v0.2.1  
- Add batch/single inference feature  

v0.2.0  
- CrossEntropyLoss ignore whitespace (prevent imbalance label)
- half-precision with amp applied  
- Save model state with model config and ngram config (easy to reload from ckpt)  
- preprocess script support multi-processing  
- remove validation step process as large corpus is available (only train acc measured)  

v0.1.0  
- n-gram tokenizer
- LMDB dataset and dataloader
- Pytorch Lightining trainer
- preprocessed Ratings Dataset