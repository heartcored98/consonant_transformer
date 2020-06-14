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


class ExampleBuilder(object):
    """Given a stream of input text, creates pretraining examples."""

    def __init__(self, tokenizer, max_length):
        self._tokenizer = tokenizer
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._target_length = max_length

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = line.strip().replace("\n", " ")
        if (not line) and self._current_length != 0:  # empty lines separate docs
            return self._create_example()
        bert_tokens = self._tokenizer.tokenize(line)
        bert_tokids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
        self._current_sentences.append(bert_tokids)
        self._current_length += len(bert_tokids)
        if self._current_length >= self._target_length:
            return self._create_example()
        return None

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self._target_length - 3) // 2

        first_segment = []
        second_segment = []
        for sentence in self._current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (len(first_segment) == 0 or
                len(first_segment) + len(sentence) < first_segment_target_length or
                (len(second_segment) == 0 and
                len(first_segment) < first_segment_target_length and
                random.random() < 0.5)):
                first_segment += sentence
            else:
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self._max_length - 2]
        second_segment = second_segment[:max(0, self._max_length -
                                                len(first_segment) - 3)]

        # prepare to start building the next example
        self._current_sentences = []
        self._current_length = 0
        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        return self._make_example(first_segment, second_segment)

    def _make_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a tf.train.Example."""
        CLS_ID = self._tokenizer.convert_tokens_to_ids('[CLS]')
        SEP_ID = self._tokenizer.convert_tokens_to_ids('[SEP]')

        input_ids = [CLS_ID] + first_segment + [SEP_ID]
        segment_ids = [0] * len(input_ids)
        if second_segment:
            input_ids += second_segment + [SEP_ID]
            segment_ids += [1] * (len(second_segment) + 1)
        input_mask = [1] * len(input_ids)
        input_ids += [0] * (self._max_length - len(input_ids))
        input_mask += [0] * (self._max_length - len(input_mask))
        segment_ids += [0] * (self._max_length - len(segment_ids))

        example = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids
        }
        return example
    

class ExampleWriter(object):
    """Writes pre-training examples to disk."""

    def __init__(self, job_id, output_dir, max_char_length,
                num_jobs, blanks_separate_docs, tokenizer,
                max_buffer_size=20000):
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
        if isinstance(input_corpus, str):
            with open(input_corpus, 'r') as f:
                lines = f.readlines()
        elif isinstance(input_corpus, list):
            lines = input_corpus

        for line in tqdm(lines):
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


def write_examples(job_id, args): # Distribute N split_data files per job
    """A single process creating and writing out pre-processed examples."""
    wiki_dir = os.path.join(args.data_dir, "wiki_split_dataset")

    def log(*args):
        msg = " ".join(map(str, args))
        print("Job {}:".format(job_id), msg)

    log("Creating example writer")
    example_writer = ExampleWriter(
        job_id=job_id,
        output_dir=os.path.join(args.data_dir, args.output_dir),
        model_size= args.model_size,
        max_seq_length=args.max_seq_length,
        num_jobs=args.num_processes,
        blanks_separate_docs=False,
        do_lower_case=args.do_lower_case
    )

    log("Writing wiki examples")
    fnames = os.listdir(wiki_dir) # 14 files produced by split of wiki_raw_data.txt
    print(fnames[:10])
    fnames = [f for (i, f) in enumerate(fnames) if i % args.num_processes == job_id] # Distribute wiki_files by processor job_id
    random.shuffle(fnames)
    start_time = time.time()
    for file_no, fname in enumerate(fnames):
        # Logging tar.gz file processing status
        if file_no > 0 and file_no % 10 == 0:
            elapsed = time.time() - start_time
            log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, ".format(
                    file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed),
                    int((len(fnames) - file_no) / (file_no / elapsed))))

        # write txt files from each of the split data file (i.e. fname) into wiki_dir
        example_writer.write_examples(os.path.join(wiki_dir, fname))
    example_writer.finish()
    
    log("Done!")
    


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/home/jovyan/dingbro/consonant_transformer/dataset",
                        help="Location of data (vocab file, corpus, etc).")
    parser.add_argument("--input-file", default="raw_ratings.txt", type=str,
                        help="Location of data (vocab file, corpus, etc).")
    parser.add_argument("--output-dir-prefix", default="ratings", type=str,
                        help="Location of data (vocab file, corpus, etc).")
    parser.add_argument("--ngram", default=3, type=int,
                        help="Number of n-gram for consonant tuples")
    parser.add_argument("--max-char-length", default=100, type=int,
                      help="Number of tokens per example.")
    parser.add_argument("--num-processes", default=4, type=int,
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
    set_seed(args)
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)

    # Split dataset into train/val 
    train_lines = lines[:int(len(lines) * 0.8)]
    val_lines = lines[int(len(lines) * 0.8):]

    print("Ngram: ", args.ngram)
    print("Max char lenght: ", args.max_char_length)
    print("Train set: ", len(train_lines), "Val set: ", len(val_lines))
                
    tokenizer = NGRAMTokenizer(args.ngram)
    print("Head2id size: ", len(tokenizer.head2id))
    print("Midtail2id size: ", len(tokenizer.midtail2id))
    example_writer = ExampleWriter(0, args.output_dir+'/train', args.max_char_length, num_jobs=1, tokenizer=tokenizer, blanks_separate_docs=False)
    example_writer.write_examples(input_corpus=train_lines)
    example_writer.finish()

    example_writer = ExampleWriter(0, args.output_dir+'/val', args.max_char_length, num_jobs=1, tokenizer=tokenizer, blanks_separate_docs=False)
    example_writer.write_examples(input_corpus=val_lines)
    example_writer.finish()
    # if args.num_processes == 1:
    #     write_examples(0, args)
    # else:
    #     jobs = []
    #     for i in range(args.num_processes):
    #         job = multiprocessing.Process(target=write_examples, args=(i, args))
    #         jobs.append(job)
    #         job.start()
    #     for job in jobs:
    #         job.join()  


if __name__ == "__main__":
    main()


# > export DATA_DIR=/home/ubuntu/CS570/dataset
# > python prepare_openwebtext.py --data-dir $DATA_DIR --num-processes 14 

# After preprocessing we could compress whole openwebtext_lmdb_128 directory with 
# > tar -zcvf openwebtext_lmdb_128.tar.gz openwebtext_lmdb_128