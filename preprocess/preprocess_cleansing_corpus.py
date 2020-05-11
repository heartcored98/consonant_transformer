import re

import pandas as pd
import kss
from tqdm import tqdm
from time import time



def clean_str(text):
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


def cleansing_df(df, min_char_length):
        
    sent_list = list()
    for i, row in tqdm(enumerate(df.iterrows())):
        """
        Ref : https://github.com/likejazz/korean-sentence-splitter
        """
        sent = str(row[1].values[0])
        clean_sent = clean_str(sent)

        if len(clean_sent) > min_char_length:
            sent_list.append(clean_sent)

    return sent_list

def save_corpus(fname, sent_list):
    with open(fname, 'w', encoding='utf-8') as output:
        for sent in tqdm(sent_list):
            output.write(sent)
            output.write('\n\n')


if __name__ == '__main__':

    ts = time()
    min_char_length = 5

    df1 = pd.read_csv('../data/raw_ratings.txt', header=None, delimiter='\t')
    df2 = pd.read_csv('../data/raw_spoken.txt', header=None, delimiter='\t')
    df3 = pd.read_csv('../data/raw_wiki_ko_sent.txt', header=None, delimiter='\t')

    sent_list = cleansing_df(df1, min_char_length) + cleansing_df(df2, min_char_length) + cleansing_df(df3, min_char_length)
    save_corpus('../data/clean_corpus.txt', sent_list)

    te = time()
    print(f"Took {te-ts} sec")


    

