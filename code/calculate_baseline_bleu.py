from collections import defaultdict
import argparse
import torch
import transformers 
import json
import os
import numpy as np
import math
import utils
import inspect
from tqdm import tqdm
import dill
from math import ceil 
import fcntl
import seaborn as sns
import sampler 
import filtering
import logging
import plotter
from scipy import stats
from nltk.translate.bleu_score import SmoothingFunction
from torch.utils.data import DataLoader
from scipy.stats import entropy

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--seed", default=os.environ.get("SEED", 5), type=int)
    parser.add_argument("--input-file", default=os.environ.get("INPUT_FILE", "input.txt"), type=str)
    parser.add_argument("--num-sentences", default=os.environ.get("NUM_SENTENCES", default=2000), type=int)
    parser.add_argument("--gram", default=os.environ.get("GRAM", 3)) 
    return parser.parse_args() 

def main(args):
    print(args)
    print("Started experiment!") 
    utils.print_args(args)
    utils.set_seed(args.seed)
    
    smoothing_method = {"nist": SmoothingFunction().method3}
    results = {}
    scores = {}
    for name, method in smoothing_method.items():
        scores[name] = utils.evaluate_bleu(args.input_file, os.path.join("data", "valid.txt"), num_real_sentences=args.num_sentences, num_generated_sentences=args.num_sentences, gram=args.gram, smoothing_method=method, chunk_size=15)
        print()

    for name in smoothing_method.keys():
        results[name] = {}
        results[name]['scores'] = scores[name]

    print("Results:", results) 
    bleu = results['nist']['scores']['bleu5']
    sbleu = results['nist']['scores']['self-bleu5']
    hmean = stats.hmean([bleu, 1.0 / sbleu])
    print("Harmonic Mean:", hmean)
    
if __name__=="__main__":
    args = parse_args()
    main(args)
