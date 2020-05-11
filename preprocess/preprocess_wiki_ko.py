import pandas as pd
import kss
import tqdm

df = pd.read_csv('../data/processed_wiki_ko.txt', header=None, delimiter='\t') 
sent_list = list()

for i, row in tqdm(enumerate(df.iterrows())):
    paragraph = str(row[1].values[0])
    sents = kss.split_sentences(paragraph)
    for j, sent in enumerate(sents):
        sent_list.append(sent)

with open('../data/processed_wiki_ko_sent.txt', 'w', encoding='utf-8') as output:
    for sent in tqdm(sent_list):
        output.write(sent)
        output.write('\n')