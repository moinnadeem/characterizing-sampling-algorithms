import argparse
from collections import defaultdict
import hashlib
from glob import glob
import os

import transformers
from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir")
    parser.add_argument("--prefix-length", default=10)
    return parser.parse_args()

def main(args):
    input_files = glob(args.input_dir + "*")
    results = Parallel(n_jobs=80)(delayed(process_file)(i) for i in input_files)
    tokens_buckets = defaultdict(lambda: [])

    for i in results:
        for k, v in i.items():
            tokens_buckets[k].extend(v) 
        
    for k, v in tokens_buckets.items():
        sampler_names = set(i[1] for i in v)
        if len(sampler_names)>=5:
            for i in v:
                print(f"{i[1]}: {i[0]}")
                print(i[2])
                print()
            print("--------------"*7)

def process_file(input_file):
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    filename = os.path.basename(input_file) 
    tokens_buckets = defaultdict(lambda: [])
    attrs = {i.split(":")[0]:i.split(":")[1] for i in filename.split("-")}
    with open(input_file) as f:
        lines = f.readlines()
    tokenized_lines = [tokenizer.tokenize(i) for i in lines]
    for idx, line in enumerate(lines):
        if attrs['sampler']=="KTemperatureSweep":
            continue
        line_id = hashlib.sha256(" ".join(tokenized_lines[idx][:args.prefix_length]).encode()).hexdigest()
        tokens_buckets[line_id].append((line, attrs['sampler'], attrs)) 
    return tokens_buckets

if __name__=="__main__":
    args = parse_args()
    main(args)
