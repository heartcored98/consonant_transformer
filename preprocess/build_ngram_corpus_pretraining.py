import argparse
import multiprocessing
import os
import sys
import random
import tarfile
import time
import shutil
from collections import defaultdict

import torch
import numpy as np
import pyxis as px
from tqdm import tqdm

sys.path.insert(0, '../')
from consonant.model.tokenization import NGRAMTokenizer
from consonant.utils import clean_str, set_seed


def rmkdir(dir_path):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    

def chunks(lst, n_chunk):
    """Yield successive n chunks from lst."""
    if len(lst) % n_chunk == 0:
        n = int(len(lst) / n_chunk)
    else:
        n = len(lst) // (n_chunk - 1)
        
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class ExampleWriter(object):
    """Writes pre-training examples to disk."""

    def __init__(self, job_id, output_dir, max_char_length,
                num_jobs, blanks_separate_docs, tokenizer,
                max_buffer_size=20000):

        self.job_id = job_id
        self._blanks_separate_docs = blanks_separate_docs
        self.max_char_length = max_char_length

        self.n_written = 0
        self.max_buffer_size = max_buffer_size
        self.buffer = defaultdict(list)

        self.tokenizer = tokenizer
        self.writer_path = os.path.join(output_dir, f"{job_id+1}_of_{num_jobs}")
        self.writer = px.Writer(dirpath=self.writer_path, map_size_limit=3000000, ram_gb_limit=10)

    def write_examples(self, input_corpus):
        """Writes out examples from the provided input file."""

        def log(*args):
            msg = " ".join(map(str, args))
            print("Job {}:".format(job_id), msg)

        if isinstance(input_corpus, str):
            with open(input_corpus, 'r') as f:
                lines = f.readlines()
        elif isinstance(input_corpus, list):
            lines = input_corpus

        for line in tqdm(lines, position=self.job_id, desc='Job ' + str(self.job_id)):
            line = line.strip()  # Remvoe return character
            if line or self._blanks_separate_docs:

                # Append line. example will return if the appended sentences reached max_char_length
                cleansed_line = clean_str(line)
                if len(cleansed_line) < 1:
                    continue
                example = self.tokenizer.encode(cleansed_line, self.max_char_length, return_attention_mask=True)
                if example:
                    self.add_example(example)

                    # If buffer exceed max_buffer_size write buffer to file system
                    if len(self.buffer['head_ids']) >= self.max_buffer_size:
                        self.flush_buffer()

    def add_example(self, example):
        for k, v in example.items():
            self.buffer[k].append(v)
        self.n_written += 1

    def flush_buffer(self):
        if len(self.buffer) > 0:
            input_ids = np.array(self.buffer['head_ids'], dtype=np.int32)
            midtail_ids = np.array(self.buffer['midtail_ids'], dtype=np.int32)
            attention_masks = np.array(self.buffer['attention_masks'], dtype=np.bool)
            self.writer.put_samples('head_ids', input_ids, 'midtail_ids', midtail_ids, 'attention_masks', attention_masks)
        self.buffer = defaultdict(list)

    def finish(self):
        self.flush_buffer()
        self.writer.close()


def write_examples(job_id, args, corpus_lines, phase): # Distribute N split_data files per job
    """A single process creating and writing out pre-processed examples."""

    def log(*args):
        msg = " ".join(map(str, args))
        print("Job {}:".format(job_id), msg)
        
    tokenizer = NGRAMTokenizer(args.ngram)
    example_writer = ExampleWriter(job_id, f"{args.output_dir}/{phase}", args.max_char_length, num_jobs=args.num_processes, tokenizer=tokenizer, blanks_separate_docs=False)
    log("Creating example writer")
    log("Writing wiki examples")

    example_writer.write_examples(input_corpus=corpus_lines)
    example_writer.finish()
    log("Done!")
    


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/home/whwodud98/consonant_transformer/dataset",
                        help="Location of data (vocab file, corpus, etc).")
    parser.add_argument("--input-file", default="news_comments_1.txt", type=str,
                        help="Location of data (vocab file, corpus, etc).")
    parser.add_argument("--output-dir-prefix", default="comments", type=str,
                        help="Location of data (vocab file, corpus, etc).")
    parser.add_argument("--ngram", default=3, type=int,
                        help="Number of n-gram for consonant tuples")
    parser.add_argument("--train-ratio", default=0.9, type=float,
                        help="train-val ratio")
    parser.add_argument("--max-char-length", default=100, type=int,
                      help="Number of tokens per example.")
    parser.add_argument("--num-processes", default=16, type=int,
                        help="Parallelize across multiple processes.")
    parser.add_argument("--seed", default=777, type=int,
                        help="Initial Seed")
    # parser.add_argument("--do-lower-case", dest='do_lower_case',
    #                     action='store_true', help="Lower case input text.")
    # parser.add_argument("--no-lower-case", dest='do_lower_case',
    #                     action='store_false', help="Don't lower case input text.")
    parser.set_defaults(do_lower_case=True)
    args = parser.parse_args()
    args.input_file = os.path.join(args.data_dir, 'raw', args.input_file)
    args.output_name = f"{args.output_dir_prefix}_{args.ngram}_{args.max_char_length}"
    args.output_dir = os.path.join(args.data_dir, 'processed', args.output_name)

    print('input', args.input_file)
    print('output', args.output_name)
    print("output dir", args.output_dir)

    if not os.path.isdir(args.output_dir):
        rmkdir(args.output_dir)
        rmkdir(args.output_dir + '/train')
        rmkdir(args.output_dir+'/val')

    # Read dataset and shuffle
    print("Starting reading file")
    set_seed(args)
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
    print(f"!! Read {len(lines)} lines !!")

    # Split dataset into train/val 
    train_lines = lines[:int(len(lines) * args.train_ratio)]
    val_lines = lines[int(len(lines) * args.train_ratio):]

    print("Ngram: ", args.ngram)
    print("Max char lenght: ", args.max_char_length)
    print("Train set: ", len(train_lines), "Val set: ", len(val_lines))

    tokenizer = NGRAMTokenizer(args.ngram)    
    print("Head2id size: ", len(tokenizer.head2id))
    print("Midtail2id size: ", len(tokenizer.midtail2id))
    # example_writer = ExampleWriter(0, args.output_dir+'/train', args.max_char_length, num_jobs=1, tokenizer=tokenizer, blanks_separate_docs=False)
    # example_writer.write_examples(input_corpus=train_lines)
    # example_writer.finish()

    # example_writer = ExampleWriter(0, args.output_dir+'/val', args.max_char_length, num_jobs=1, tokenizer=tokenizer, blanks_separate_docs=False)
    # example_writer.write_examples(input_corpus=val_lines)
    # example_writer.finish()

    phase = 'train'
    if args.num_processes == 1:
        write_examples(0, args, train_lines, phase)
    else:
        split_lines = list(chunks(train_lines, args.num_processes))
        jobs = []
        for i in range(args.num_processes):
            job = multiprocessing.Process(target=write_examples, args=(i, args, split_lines[i], phase))
            jobs.append(job)
            job.start()
        for job in jobs:
            job.join()  


    phase = 'val'
    if args.num_processes == 1:
        write_examples(0, args, val_lines, phase)
    else:
        split_lines = list(chunks(val_lines, args.num_processes))
        jobs = []
        for i in range(args.num_processes):
            job = multiprocessing.Process(target=write_examples, args=(i, args, split_lines[i], phase))
            jobs.append(job)
            job.start()
        for job in jobs:
            job.join()  


if __name__ == "__main__":
    main()


# > export DATA_DIR=/home/ubuntu/CS570/dataset
# > python prepare_openwebtext.py --data-dir $DATA_DIR --num-processes 14 

# After preprocessing we could compress whole openwebtext_lmdb_128 directory with 
# > tar -zcvf openwebtext_lmdb_128.tar.gz openwebtext_lmdb_128