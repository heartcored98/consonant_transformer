# ConsonantTransformer  

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.web.app/redirect?git_repo=https://github.com/heartcored98/consonant_transformer)


# Inference with TorchServer  

```
torch-model-archiver --model-name medium_consonant --version 1.0 --serialized-file ./deploy/model_store/medium_consonant_0189000.bin --extra-files deploy/setup_config.json --handler ./deploy/transformer_handler.py --export-path deploy/model_store
```

```
torchserve --start --ts-config ./deploy/config.properties --model-store ./deploy/model_store --models medium_consonant=medium_consonant.mar

```  

```
usr/bin/curl  http://127.0.0.1:8080/predictions/medium_consonant -T sample.txt
```


```
/usr/bin/curl -H "Content-Type: application/json"   http://127.0.0.1:8080/predictions/medium_consonant -d '{"text":"안녕하세요? 진짜 기분이 너무 너무 안 좋아 어쩌지 먹고 싶다"}'
```

### Docker Build  

```
docker build -t whwodud98/consonant:<model_tag> .
```

### Push to DockerHub

```
docker push whwodud98/consonant:<model_tag>
```

### Run Inference Docker Image    

Currently, `medium` and `extended_base` tags are supported. 

```
docker run -it -p 8080:8080 -p 8081:8081 whwodud98/consonant:<model_tag>
```

### HTTP Prediciton 
```
/usr/bin/curl -H "Content-Type: application/json"   http://ec2-13-124-68-75.ap-northeast-2.compute.amazonaws.com:8080/predictions/medium_consonant -d '{"text":"안녕하세요? 진짜 기분이 너 무 너무 안 좋아 어쩌지 먹고 싶다"}'
```   


### TO-DO  

- [ ] Update model part with transformers v4 (currently working with v2)  
- [ ] Re-train model with 모두의 말뭉치 dataset  
- [ ] Deploy to Ainize Service  
- [ ] Support partially-filled consonants. (e.g. ㅇㄴㅎㅅㅇ 오늘 하늘이 ㄴㅁ ㅇㅃㅇ!)  
- [ ] Document datset pre-processing procedure  