from itertools import product
import re

import numpy as np


class NGRAMTokenizer():

    BASE_CODE, HEAD, MID = 44032, 588, 28

    # 초성 리스트. 00 ~ 18
    HEAD_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    EXTRA_LIST = [' ', ',', '.', '?', '!', '~', '∼']

    # 중성 리스트. 00 ~ 20
    MID_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ', '@']

    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    TAIL_LIST = ['#', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', '@']

    def __init__(self, ngram, max_char_length, head_list=None, mid_list=None, tail_list=None):
        self.ngram = ngram
        self.max_char_length = max_char_length
        self.head_list = head_list if head_list else self.HEAD_LIST
        self.mid_list = mid_list if mid_list else self.MID_LIST
        self.tail_list = tail_list if tail_list else self.TAIL_LIST

        self.head2id = self.generate_head_ngram2id(self.EXTRA_LIST+self.head_list, self.ngram)
        self.mid2id = {mid:i for i,mid in enumerate(self.mid_list)}
        self.tail2id = {tail:i for i,tail in enumerate(self.tail_list)}

    def generate_head_ngram2id(self, head_list, ngram):
        """Generate all possible ngram combination of head consonant set.
        We could simply generate ngram with itertools.product function.

        Args:
            head_list (list of char): registered set of head consonants.
            ngram (int): n-gram number

        Returns:
            dict: dictionary with the key of ngram tuple and index as value.
        """
        ngram_list = list(product(head_list, repeat = ngram))
        ngram2id = {ngram_head:i for i,ngram_head in enumerate(ngram_list)}
        return ngram2id

    def encode(self, sent_list):
        """Encode list of the sentence and return batch of head_token_ids, mid_token_ids, and tail_token_ids.

        Args:
            sent_list (list of str): [description]

        Returns:
            list: [description]
        """
        encoded_sent_list = list()

        list_head_ids = list()
        list_mid_ids = list()
        list_tail_ids = list()

        if isinstance(sent_list, str):
            return self.encode_sent(sent_list)

        for sent in sent_list:
            head_ids, mid_ids, tail_ids = self.encode_sent(sent)
            list_head_ids.append(head_ids)
            list_mid_ids.append(mid_ids)
            list_tail_ids.append(tail_ids)

        return list_head_ids, list_mid_ids, list_tail_ids

    def encode_sent(self, sent):
        """Encode single sentence and return head_token_ids, mid_token_ids, and tail_token_ids.
        consonant seperation code is based on https://github.com/neotune/python-korean-handler/blob/master/korean_handler.py

        Args:
            sent (str): sentence to tokenize

        Returns:
            list: [description]
        """
        heads = list()
        mids = list()
        tails = list()

        for i, keyword in enumerate(sent[:self.max_char_length]): # truncate with max_char_length
            # 한글 여부 check 후 분리
            if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
                char_code = ord(keyword) - self.BASE_CODE
                head = int(char_code / self.HEAD)
                heads.append(self.head_list[head])

                mid = int((char_code - (self.HEAD * head)) / self.MID)
                mids.append(self.mid_list[mid])

                tail = int((char_code - (self.HEAD * head) - (self.MID * mid)))
                tails.append(self.tail_list[tail])
            else: # For non-korean character, put character on initial consonant and put @ for mid/last consonant.
                heads.append(keyword)
                mids.append('@')
                tails.append('@')

        # Initialize token ids with zero vectors
        head_ids = np.zeros(self.max_char_length, dtype=np.int)
        mid_ids = np.zeros(self.max_char_length, dtype=np.int)
        tail_ids = np.zeros(self.max_char_length, dtype=np.int)

        # Calculate left, right offset
        if self.ngram % 2 == 0: # even ngram
            left_offset = (self.ngram) // 2
            right_offset = (self.ngram-1) // 2
        else: # odd ngram
            left_offset = (self.ngram-1) // 2
            right_offset = (self.ngram-1) // 2

        # Convert consonant to id
        for i, (head, mid, tail) in enumerate(zip(heads, mids, tails)):
            ngram = heads[max(i-left_offset, 0):min(i+right_offset+1, len(heads))]
            if i < left_offset: # pad space on right side if needs
                margin = left_offset - i
                ngram = [' '] * margin + ngram
            if (len(heads)-1-i) >= 0: # pad space on left side if needs
                margin = right_offset - (len(heads)-1-i)
                ngram = ngram + [' '] * margin 

            ngram = tuple(ngram)
            head_ids[i] = self.head2id[ngram] + 1  # plus 1 for not using token
            mid_ids[i] = self.mid2id[mid] + 1
            tail_ids[i] = self.tail2id[tail] + 1

        return head_ids, mid_ids, tail_ids
            

if __name__ == '__main__':
    sentence = ["내가 너 엄청 좋아해!"]
    tokenizer = NGRAMTokenizer(3, 15)

    print("Num Head Vocab:", len(tokenizer.head2id))
    print("Num  Mid Vocab:", len(tokenizer.mid2id))
    print("Num Tail Vocab:", len(tokenizer.tail2id))

    head_ids, mid_ids, tail_ids = tokenizer.encode(sentence)
    print(head_ids, mid_ids, tail_ids)
    print("Head Consonant ID")
    print('->', head_ids[0])

    print()
    print("Mid Consonant ID")
    print('->', mid_ids[0])

    print()
    print("Tail Consonant ID")
    print('->', tail_ids[0])
