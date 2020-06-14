import re
import random
import numpy as np
import torch


def clean_str(text):
    """
    [Ref1] : https://data-newbie.tistory.com/210  
    [Ref2] : 
    """
    #print("Original Sentence:", text)

    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("Email removed:", text)

    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # http / ftp / https
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '(www).(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # www url 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("URL removed:", text)


    pattern = '<[^>]*>' # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("HTML tag removed:", text)

    pattern = '\([^)]*\)'
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '\[[^)]*\]'
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("Word in Parathesis", text)

    pattern = re.compile(r'[^ .,?!~∼가-힣]+')  # Remove alphabetic, parathesis, only consonant characters
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("Alphabetic, Speical character removed:", text)

    text = " ".join(text.split()) # Remove redundant whitespace
    #print("Whitespace removed:", text)

    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    #print("punctuation with whitespace removed:", text)

    return text


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

if __name__ == '__main__':
    sent = "하이하이 ㅇㄴ은 너무 기분 조아!@@"
    print(clean_str(sent))

    # > 하이하이 은 너무 기분 조아!
