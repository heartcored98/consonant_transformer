from itertools import product
import re

import numpy as np
# import torch

from consonant.model.jamo import join_jamos


class NGRAMTokenizer():

    BASE_CODE, HEAD, MID = 44032, 588, 28
    UNKNOWN_HEAD = '@'

    # 초성 리스트. 00 ~ 18
    HEAD_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    EXTRA_LIST = [' ', ',', '.', '?', '!', '~', UNKNOWN_HEAD]

    # 중성 리스트. 00 ~ 20
    MID_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    TAIL_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    def __init__(self, ngram, head_list=None, mid_list=None, tail_list=None):
        self.ngram = ngram
        self.head_list = head_list if head_list else self.HEAD_LIST
        self.mid_list = mid_list if mid_list else self.MID_LIST
        self.tail_list = tail_list if tail_list else self.TAIL_LIST

        self.head2id, self.id2head = self.generate_head_ngram2id(self.EXTRA_LIST + self.head_list, self.ngram)
        self.midtail2id, self.id2midtail = self.generate_midtail2id(self.mid_list, self.tail_list)

    def generate_head_ngram2id(self, head_list, ngram):
        """Generate all possible ngram combination of head consonant set.
        We could simply generate ngram with itertools.product function.

        Args:
            head_list (list of char): registered set of head consonants.
            ngram (int): n-gram number

        Returns:
            dict: dictionary with the key of ngram tuple and index as value.
        """
        ngram_list = ['[PAD]', '[CLS]', '[SEP]'] + list(product(head_list, repeat = ngram))

        ngram2id = {ngram_head: i for i, ngram_head in enumerate(ngram_list)}
        id2ngram = {i: ngram_head for i, ngram_head in enumerate(ngram_list)}
        return ngram2id, id2ngram

    def generate_midtail2id(self, mid_list, tail_list):
        """Generate all possible mid-tail consonant combination.

        Args:
            mid_list ([type]): [description]
            tail_list ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        midtail_list = [''] + list(product(mid_list, tail_list))
        midtail2id = {midtail: i for i, midtail in enumerate(midtail_list)}
        id2midtail = {i: midtail for i, midtail in enumerate(midtail_list)}
        return midtail2id, id2midtail

    def replace_unknown_head(self, heads):
        heads = list(heads)
        for i, char in enumerate(heads):
            if char not in self.head_list + self.EXTRA_LIST:
                heads[i] = self.UNKNOWN_HEAD
        return ''.join(heads)

    def encode(self, sent_list, max_char_length=128, return_tensors=None, return_attention_mask=None):
        """Encode list of the sentence and return dictionary of head_token_ids, mid_token_ids, and tail_token_ids batch.

        Args:
            sent_list (list of str): [description]
            return_tensors (None or 'pt'): Return numpy matrix if None. Return torch tensor if 'pt'

        Returns:
            list: [description]
        """

        def stack(encoded_sent, key, return_tensors):
            output = [item[key] for item in encoded_sent]
            return torch.Tensor(output, dtype=torch.int32) if return_tensors == 'pt' else np.vstack(output)

        if isinstance(sent_list, str):
            return self.encode_sent(sent_list, max_char_length, return_attention_mask)
        
        encoded_sent = list()
        for sent in sent_list:
            encoded_sent.append(self.encode_sent(sent, max_char_length, return_attention_mask))
            
        output = {
            'head_ids': stack(encoded_sent, 'head_ids', return_tensors),
            'midtail_ids': stack(encoded_sent, 'midtail_ids', return_tensors),
        }
        if return_attention_mask:
            output.update(
                {'attention_masks': stack(encoded_sent, 'attention_masks', return_tensors)}
            )

        return output

    def encode_sent(self, sent, max_char_length, return_attention_mask):
        """Encode single sentence and return head_token_ids, mid_token_ids, and tail_token_ids.
        consonant seperation code is based on https://github.com/neotune/python-korean-handler/blob/master/korean_handler.py

        Args:
            sent (str): sentence to tokenize

        Returns:
            list: [description]
        """
        heads = list()
        midtails = list()

        for i, keyword in enumerate(sent[:max_char_length-2]): # truncate with max_char_length-2 due to [CLS] and [SEP] tokens
            # 한글 여부 check 후 분리
            if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
                char_code = ord(keyword) - self.BASE_CODE
                head = int(char_code / self.HEAD)
                heads.append(self.head_list[head])

                mid = int((char_code - (self.HEAD * head)) / self.MID)
                tail = int((char_code - (self.HEAD * head) - (self.MID * mid)))
                midtails.append((self.mid_list[mid], self.tail_list[tail]))

            else: # For non-korean character, put character on initial consonant and put @ for mid/last consonant.
                heads.append(keyword)
                midtails.append('')

        # Initialize token ids with [PAD] tokens
        head_ids = np.zeros(max_char_length, dtype=np.int)
        midtail_ids = np.zeros(max_char_length, dtype=np.int)
        midtail_ids.fill(-100)

        # Calculate left, right offset
        if self.ngram % 2 == 0: # even ngram
            left_offset = (self.ngram) // 2
            right_offset = (self.ngram-1) // 2
        else: # odd ngram
            left_offset = (self.ngram-1) // 2
            right_offset = (self.ngram-1) // 2

        head_ids[0] = self.head2id['[CLS]'] # Add CLS token at the begining of the sentence
        # Convert consonant to id
        for i, (head, midtail) in enumerate(zip(heads, midtails)):
            ngram = heads[max(i-left_offset, 0):min(i+right_offset+1, len(heads))]
            if i < left_offset: # pad space on right side if needs
                margin = left_offset - i
                ngram = [' '] * margin + ngram
            if (len(heads)-1-i) >= 0: # pad space on left side if needs
                margin = right_offset - (len(heads)-1-i)
                ngram = ngram + [' '] * margin 

            ngram = self.replace_unknown_head(ngram)
            ngram = tuple(ngram)
            head_ids[i+1] = self.head2id[ngram] if ngram in self.head2id else self.head2id['[UNK]'] # look up n-gram otherwise Unknown token
            midtail_ids[i+1] = self.midtail2id[midtail] 
        head_ids[i+2] = self.head2id['[SEP]'] # Add seperation token at the end of the sentence

        output = {
            'head_ids': head_ids,
            'midtail_ids': midtail_ids,
        }

        if return_attention_mask:
            attention_mask = np.hstack([np.ones(len(heads)+2, dtype=np.int), np.zeros(max_char_length - len(heads) - 2, dtype=np.int)])
            output.update(
                {'attention_masks': attention_mask}
            )
        return output
            

    def decode(self, head_ids, midtail_ids):
        """Decode the original sentence from the list of ids
        """
        for (head_ids, midtail_ids) in zip(head_ids, midtail_ids):
            for head, midtail in zip():
                pass
            print(head_ids, midtail_ids)

    def decode_sent(self, head_ids, midtail_ids):
        position = int(self.ngram / 2)

        sent = ""
        for head, midtail in zip(head_ids, midtail_ids):
            if head in [0, 1, 2, 3]:
                continue
            sent += join_jamos(self.id2head[head][position] + ''.join(self.id2midtail[midtail]))

        return sent
    

if __name__ == '__main__':
    sentence = ["각나?", "힣"]
    tokenizer = NGRAMTokenizer(4)

    print("Num Head Vocab:", len(tokenizer.head2id))
    print("Num  Mid Vocab:", len(tokenizer.midtail2id))

    result = tokenizer.encode(sentence, max_char_length=30, return_attention_mask=True) #, return_tensors='pt')
    head_ids = result['head_ids']
    midtail_ids = result['midtail_ids']
    attention_masks = result['attention_masks']

    print(head_ids, midtail_ids)
    print("Head Consonant ID")
    print('->', head_ids[0])

    print()
    print("Midtail Consonant ID")
    print('->', midtail_ids[0])

    print()
    print("Attention Mask")
    print('->', attention_masks[0])

    print("=========================")

    result = tokenizer.decode_sent(head_ids[0], midtail_ids[0])

    print(tokenizer.replace_unknown_head('ㄱㄴ!^ㅎ'))
    print(tokenizer.id2midtail[0])