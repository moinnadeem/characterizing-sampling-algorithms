from colorama import Back, Fore, Style, init
import metrics

import torch
import os
import sys
from time import sleep
import json
import logging
import numpy as np
import random
from nltk.tokenize import sent_tokenize
import sys

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def print_args(args):
    print("---------------------------------------------------------------")
    print(f"{Fore.LIGHTCYAN_EX}                     ARGUMENTS                 {Style.RESET_ALL}")
    for arg in vars(args):
        name = " ".join([i.capitalize() for i in arg.split("_")])
        print(f"{Fore.LIGHTCYAN_EX}{name}:{Style.RESET_ALL} {getattr(args, arg)}")
    print("---------------------------------------------------------------")
    print()

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def evaluate_bleu(generated_text_path, real_text_path, num_real_sentences, num_generated_sentences, gram=5, smoothing_method=None, chunk_size=10):
    if smoothing_method==None:
        raise Exception("Must specify a smoothing method")
    print("Computing bleu score...")
    bleu5 = metrics.Bleu(test_text=generated_text_path,
        real_text=real_text_path,
        num_real_sentences=num_real_sentences,
        num_fake_sentences=num_generated_sentences,
        smoothing_method=smoothing_method,
        gram=gram,
        chunk_size=chunk_size).get_score()
    print("Computing self-BLEU score...")
    sbleu5 = metrics.SelfBleu(test_text=generated_text_path,
        real_text=real_text_path,
        num_real_sentences=num_generated_sentences,
        num_fake_sentences=num_generated_sentences,
        smoothing_method=smoothing_method,
        gram=gram,
        chunk_size=chunk_size).get_score()
    return {"bleu5": bleu5, "self-bleu5": sbleu5}

def evaluate_nist(generated_text_path, real_text_path, num_real_sentences, num_generated_sentences, gram=5):
    print("Computing NIST score...")
    nist5 = metrics.Nist(test_text=generated_text_path,
        real_text=real_text_path,
        num_real_sentences=num_real_sentences,
        num_fake_sentences=num_generated_sentences,
        gram=gram).get_score(is_fast=False)
    print("Computing self-NIST score...")
    snist5 = metrics.SelfNist(test_text=generated_text_path,
        num_real_sentences=num_real_sentences,
        num_fake_sentences=num_generated_sentences,
        gram=gram).get_score()
    return {"nist": nist5, "self-nist": snist5}


def evaluate(desired_metrics, generated_text_path, real_text_path, num_real_sentences, num_generated_sentences, gram=5):
    results = {}
    for metric in desired_metrics:
        print(f"Running {metric}")
        result = getattr(metrics, metric)(test_text=generated_text_path,
            real_text=real_text_path,
            num_real_sentences=num_real_sentences,
            num_fake_sentences=num_generated_sentences,
            gram=gram).get_score()
        results[metric] = result
    return results

def write_sentences(sentences, output_file):
    """
    Invariant: files will have </s>, so decode them upon reading.
    """
    with open(output_file, "w+") as f:
        for sentence in sentences:
            sentence = "</s>".join(sentence.split("\n"))
            f.write(sentence + "\n")

def preprocess_text(lines, tokenizer, lmin=40, lmax=50, remove_prefix_length=None):
    """
    Processes text to the nearest sentence within min, max bounds.
    """
    truncated_lines = []
    filtered_indicies = []  # keep track of the indicies we keep!
    lengths = []  # keep track of how many subwords are in a sentence
    for idx, line in enumerate(lines):
        # if "= = =" in line:
            # # let's not ask people to rate quality for headers?
            # continue

        # let's do some tokenization cleaning!
        line = line.replace("</s>", "\n").replace(" @", "").replace("@ ", "").strip()
        line = tokenizer.clean_up_tokenization(line)
        tokens = sent_tokenize(line)
        truncated = ""
        for sentence in tokens:
            # find the rightmost index of the token (rindex + len(token))
            end_pos = line.rindex(sentence) + len(sentence)

            # extend sentence until there
            truncated = line[:end_pos]
            # print(sentence)
            # print(truncated)
            # input("Wait.")
            model_tokens = tokenizer.tokenize(truncated)

            # if sentence in range of [min, max], add it to the list
            # and break out of the loop
            if len(model_tokens) >= lmin and len(model_tokens) <= lmax:
                processed = truncated
                # postprocess
                if remove_prefix_length is not None and remove_prefix_length > 0:
                    token = tokenizer.convert_tokens_to_string(model_tokens[:remove_prefix_length]) 
                    sindex = processed.rfind(token)
                    if sindex < 0:
                        print(f"AN ERROR OCCURED: could not find '{token}' in '{processed}'")
                    context =  sindex + len(token)
                    processed = processed[context:]

                processed = processed.replace("\n", "</s>")
                truncated_lines.append(processed)
                filtered_indicies.append(idx)
                lengths.append(len(model_tokens))
                break
    assert len(truncated_lines)==len(filtered_indicies)
    assert len(lengths)==len(filtered_indicies)
    return truncated_lines, filtered_indicies, lengths 

def calculate_logprobs(input_tokens, filtered_scores, original_scores, prefix_length, filter_value, interpolate_ratio=0.9, batch_size=500):
    # input_tokens 101 (prefix+bos) 
    # scores: 89 (10 for prefix, 1 for the BOS)
    assert filter_value is not None
    assert prefix_length is not None

    idxs = (input_tokens==13)
    eot_index = torch.max(idxs, dim=1)[1] + 1
    input_tokens = input_tokens[:, prefix_length + 1:].unsqueeze(-1)
    original_scores = original_scores[:, prefix_length:]
    filtered_scores = filtered_scores.softmax(dim=-1)
    original_scores = original_scores.softmax(dim=-1)
    # print(filtered_scores)
    # print(filtered_scores.nonzero())

    for batch in range(0, filtered_scores.shape[0], batch_size):
        # zeros_mask = filtered_scores[batch:batch+BATCH_SIZE]==0
        # filtered_scores[batch:batch+BATCH_SIZE][zeros_mask] = 1 
        # filtered_scores[batch:batch+BATCH_SIZE] = filtered_scores[batch:batch+BATCH_SIZE].log()
        # # filtered_scores[zeros_mask] = (filtered_scores[zeros_mask]).log()
        # zeros_mask = original_scores[batch:batch+BATCH_SIZE]==0
        # original_scores[batch:batch+BATCH_SIZE][zeros_mask] = 1
        # original_scores[batch:batch+BATCH_SIZE] = original_scores[batch:batch+BATCH_SIZE].log()

        zeros_mask = filtered_scores[batch:batch+batch_size]!=0
        filtered_scores[batch:batch+batch_size][zeros_mask] = torch.log(filtered_scores[batch:batch+batch_size][zeros_mask])
        zeros_mask = original_scores[batch:batch+batch_size]!=0
        original_scores[batch:batch+batch_size][zeros_mask] = torch.log(original_scores[batch:batch+batch_size][zeros_mask])

    scores = (interpolate_ratio * filtered_scores) + ((1.0 - interpolate_ratio) * original_scores)
    # scores = original_scores 
    
    for idx in range(eot_index.shape[0]):
        scores[idx, eot_index[idx].item():] = 0.0

    zeros_mask = scores!=0
    # scores[zeros_mask] = torch.log(scores[zeros_mask])
    # print(scores)
    # print(scores.nonzero())
    # sys.exit()
    per_item_scores = torch.gather(scores, 2, input_tokens)

    return per_item_scores.squeeze(dim=-1).sum(dim=-1)

def chunk_and_prefix_file(reference_file, tokenizer, lmin, lmax, dest_file, prefix_length=None):
    """
    Warning: adds a new token to the tokenizer!
    """
    with open(reference_file, "r+") as f:
        lines = f.readlines()

    # add tokens to tokenizer
    num_tokens = tokenizer.add_special_tokens({"additional_special_tokens": ["</s>"]})
    print(f"Added {num_tokens} tokens!")
    truncated_lines, filtered_indicies, lengths = preprocess_text(lines, tokenizer, lmin=lmin, lmax=lmax, remove_prefix_length=prefix_length)

    with open(dest_file, "w+") as f:
        for line in truncated_lines:
            f.write(line)
            f.write("\n")

class DuplicateFilter(logging.Filter):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

def lock(f):
    lock = f + ".lock" 
    unlock = f + ".unlock"
    open(unlock, "w+").close()
    print("Looking for file lock:", lock)

    while True:
        if os.path.exists(lock):
            sleep(1) 
        else:
            try:
                os.rename(unlock, lock)
                break
            except:
                pass

def unlock(f):
    lock = f + ".lock" 
    unlock = f + ".unlock"
    os.rename(lock, unlock)
