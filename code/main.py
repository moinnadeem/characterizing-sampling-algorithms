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
from time import sleep
import random
import string
import seaborn as sns
import sampler
import filtering
import logging
import plotter
from filelock import SoftFileLock 
from scipy import stats
from nltk.translate.bleu_score import SmoothingFunction
from torch.utils.data import DataLoader
from scipy.stats import entropy
from score import score_gold
import embeddings
import hashlib
import gc
import sys

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
# we add a filter to suppress duplicate log messages
duplicate_filter = utils.DuplicateFilter()
logger.addFilter(duplicate_filter)
transformers_logger = logging.getLogger("transformers.tokenization_utils")
transformers_logger.addFilter(duplicate_filter)

def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

def parse_args():
    evaluation_choices = ["Embedding", "BLEU"]

    parser = argparse.ArgumentParser() 
    parser.add_argument("--pretrained-class", default=os.environ.get("PRETRAINED_CLASS", "models/gigaword_gpt2"))
    parser.add_argument("--ctrl-code", default=os.environ.get("CTRL_CODE", "Links"), type=str)
    parser.add_argument("--seed", default=os.environ.get("SEED", 5), type=int)
    parser.add_argument("--gram", default=os.environ.get("GRAM", 3), type=int) 
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "WARNING"), choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']) 
    parser.add_argument("--max-seq-length", default=os.environ.get("MAX_SEQ_LENGTH", 100), type=int)
    parser.add_argument("--temperature", default=os.environ.get("TEMPERATURE", 1.0), type=float)
    parser.add_argument("--generation-batch-size", default=os.environ.get("GENERATION_BATCH_SIZE", 1000), type=int)
    parser.add_argument("--encoding-batch-size", default=os.environ.get("ENCODING_BATCH_SIZE", 500), type=int)
    parser.add_argument("--overwrite-cache", default=os.environ.get("OVERWRITE_CACHE", 0), type=int, help="Overwrite tokenization cache.") 
    parser.add_argument("--num-sentences", default=os.environ.get("NUM_SENTENCES", default=6000), type=int)
    parser.add_argument("--dry-run", default=os.environ.get("DRY_RUN", default=0), type=int)
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", ""), type=str)
    parser.add_argument("--prefix-file", default=os.environ.get("PREFIX_FILE", "data/gigaword/test.txt_filtered"), type=str)
    parser.add_argument("--prefix-length", default=os.environ.get("PREFIX_LENGTH", 10), type=int)
    parser.add_argument("--results-file", default=os.environ.get("RESULTS_FILE", "results.json"), type=str)
    parser.add_argument("--plot-gold", type=int, default=os.environ.get("PLOT_GOLD", 0))
    parser.add_argument("--eval-text", default=os.environ.get("EVAL_TEXT", "data/gigaword/test.txt_filtered"), type=str)
    parser.add_argument("--eval-method", default=os.environ.get("EVAL_METHOD", "BLEU"), type=str, choices=evaluation_choices)
    parser.add_argument("--knn", type=int, default=os.environ.get("KNN", 10)) 
    parser.add_argument("--preprocessed-min", type=int, default=os.environ.get("PREPROCESSED_MIN", 40)) 
    parser.add_argument("--preprocessed-max", type=int, default=os.environ.get("PREPROCESSED_MAX", 50)) 
    parser.add_argument("--filter-weight", type=int, default=os.environ.get("FILTER_WEIGHT", 0.9)) 
    
    subparser = parser.add_subparsers(help="Choose a sampler.", dest="sampler") 
    subparsers = {}
    samplers = all_subclasses(sampler.Sampler)

    for sched in samplers:
        sampler_name = sched.__name__
        subparsers[sampler_name] = subparser.add_parser(sampler_name, help=f"Sampler for {sched.__name__}")
        for k, _ in inspect.signature(sched).parameters.items():
            if k=="self":
                continue
            environ_name = k.upper()
            if k=="is_top_p":
                subparsers[sampler_name].add_argument(f"--{k}", default=os.environ.get(environ_name, False)) 
            else:
                subparsers[sampler_name].add_argument(f"--{k}", default=os.environ.get(environ_name, None)) 
            
    return parser.parse_args(), subparsers 

def main(args, subparsers):
    print(args)
    print("Started experiment!") 
    utils.print_args(args)
    utils.set_seed(args.seed)
    
    ###################################################################################
    ################################# Intialization ################################### 
    ###################################################################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(args.pretrained_class, local_files_only=True).eval()
    # register special tokens
    # num_added_tokens = tokenizer.add_special_tokens({"bos_token": "<BOS>", "eos_token": "<EOS>", 
                                                    # "pad_token": "<PAD>"}) 
    model.to(device) 

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.pretrained_class, local_files_only=True)
    sampler_args = vars(subparsers[args.sampler].parse_known_args()[0])
    sampler_args_items = "-".join([f"{k}:{v}" for k, v in sampler_args.items()])

    tokenizer_args = f"tokenizer:{tokenizer.__class__.__name__}"
    args.pretrained_class = args.pretrained_class.replace("/", "_")
    if args.pretrained_class=="ctrl":
        tokenizer_args += f"-ctrl_code:{args.ctrl_code}"
    elif "gpt2" in args.pretrained_class:
        tokenizer.pad_token = tokenizer.eos_token

    pretrained_class = args.pretrained_class.replace("-", "_")
    sampler_name = args.sampler
    if (args.sampler=="NegativeSampler"):
        print(sampler_args)
        sampler_name += "_Negative_" + sampler_args['negative_base']
    output_file = f"model:{model.__class__.__name__}-model_class:{pretrained_class}-{tokenizer_args}-sampler:{sampler_name}-temperature:{args.temperature}-seq_length:{args.max_seq_length}-ngram:{args.gram}-{sampler_args_items}.txt"

    results_file = os.path.join("results/", args.pretrained_class, args.results_file)
    
    # if our results file eixsts
    if os.path.exists(results_file):
        with open(results_file, "r+") as f:
            current = json.load(f)
        key = output_file[:-4]
        # check if we have already ran this
        if key in current:
            raise Exception("We've already computed the result!" + " " + results_file)
    
    print("Using", args.prefix_file, "as the prefix file!")
    if not args.prefix_file:
        if args.pretrained_class=="ctrl":
            input_tokens = [tokenizer.control_codes[args.ctrl_code]]
        else:
            input_tokens = [tokenizer.bos_token_id]
        input_tokens = torch.tensor(input_tokens).to(device).unsqueeze(0)
    else:
        with open(args.prefix_file, "r") as f:
            # remove lines that are empty
            lines = [] 
            for line in f.readlines():
                if line.strip() and line.count(" ") > args.prefix_length:
                    lines.append(line)

            # shuffle to ensure we have some diversity
            random.shuffle(lines)
            # truncate to number of the sentences that we are generating
            lines = lines[:args.num_sentences]
            input_tokens = tokenizer.batch_encode_plus(lines, add_special_tokens=False, truncation=True, 
                    max_length=args.prefix_length, padding="max_length", return_tensors="pt")
            attention_mask = input_tokens['attention_mask']
            input_tokens = input_tokens['input_ids']
            attn_token = torch.tensor([1]).unsqueeze(0).repeat(args.num_sentences, 1)
            attention_mask = torch.cat((attn_token, attention_mask), dim=1)
            assert tokenizer.bos_token_id not in input_tokens[0]
            bos_token = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).repeat(args.num_sentences, 1)
            input_tokens = torch.cat((bos_token, input_tokens), dim=1)

    print("Input Tokens:", input_tokens.shape)

    all_sentences = []
    k_primes, p_primes, entropy_primes = [], [], []
    num_sentences_left = args.num_sentences
    sentences_per_batch = args.generation_batch_size 
    all_logprobs = []
    
    with torch.no_grad():
        for idx in range(ceil(args.num_sentences / sentences_per_batch)):
            batch_size = None
            if num_sentences_left > sentences_per_batch:
                batch_size = sentences_per_batch
            else:
                batch_size = num_sentences_left

            schedule = getattr(sampler, args.sampler)(**sampler_args)
            if input_tokens.shape[0]==1:
                num_return_sequences = 1 
                input_ids = input_tokens
            else:
                input_ids = input_tokens[idx:idx+batch_size].to(device)
                num_return_sequences = 1 

            num_sentences_left -= batch_size

            sentences, model_logits, transformed_logits = filtering.generate(
                model=model,
                input_ids=input_ids, 
                max_length=args.max_seq_length,
                do_sample=True,
                num_beams=None,
                temperature=args.temperature,
                schedule=schedule,
                repetition_penalty=1.0,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_return_sequences,
                dry_run=args.dry_run
            )

            #########################################################################
            ############################### K Prime #################################
            #########################################################################
            sz = list(transformed_logits.size()) 
            mask = (sentences[:, -sz[1]:] > 0).cuda() #careful! make sure this mask makes sense

            distro = torch.softmax(transformed_logits.view(-1, sz[-1]).cuda(), dim = -1)
            #use .float() for Bool to avoid bug!!!
            k_prime = torch.sum((distro > (1.0 / transformed_logits.size(-1))).float(), dim = -1).view(sz[0], sz[1])
            k_prime = torch.masked_select(k_prime, mask)
            assert(torch.min(k_prime).item() > 0)
            k_prime = torch.log(k_prime)
            
            k_primes.extend(k_prime.cpu().tolist())
            #print('k_primes:', np.mean(k_primes))
            e_distro = torch.softmax(model_logits.contiguous().view(-1, sz[-1]).cuda(), dim = -1)

            ori_distro = torch.softmax(model_logits[:, -sz[1]:, :].contiguous().view(-1, sz[-1]).cuda(), dim = -1)
            
            distro = torch.softmax(transformed_logits.view(-1, sz[-1]).cuda(), dim = -1)

            ori_distro = ori_distro * (distro > (1.0 / transformed_logits.size(-1)))
            p_prime = torch.sum(ori_distro, dim = -1).view(sz[0], sz[1])
            p_prime = torch.log(torch.masked_select(p_prime, mask).float())
            p_primes.extend(p_prime.cpu().tolist())
            
            distro = torch.softmax(transformed_logits.view(-1, sz[-1]).cuda(), dim = -1)
            entropy = - torch.sum(distro * torch.log(distro + 1e-10), dim = - 1).view(sz[0], sz[1])
            entropy = torch.masked_select(entropy, mask)
            entropy_primes.extend(entropy.cpu().tolist())

            ##################################################################
            ############################ K Prime Ends ##########################
            ##################################################################
             
            transformed_logits = transformed_logits.to(device)
            model_logits = model_logits.to(device)
            sentences = sentences.to(device)
            logprobs = utils.calculate_logprobs(sentences, transformed_logits, model_logits, args.prefix_length, 0, interpolate_ratio=args.filter_weight, batch_size=args.generation_batch_size) 
            del model_logits
            del transformed_logits
            gc.collect()

            all_logprobs.append(logprobs.cpu().detach())
            all_sentences.append(sentences.cpu().detach())
            
    all_sentences = torch.cat(all_sentences, dim=0)
    all_logprobs = torch.cat(all_logprobs, dim=0)
    k_prime, p_prime, entropy_prime = np.mean(k_primes), np.mean(p_primes), np.mean(entropy_primes)
    print('Entropy Prime:', entropy_prime, 'K Prime:', k_prime, 'P Prime:', p_prime)
    results = {'k_prime': k_prime, 'p_prime': p_prime, 'entropy_prime': entropy_prime}
   
    del model
    print("Final shapes:", all_sentences.shape, all_logprobs.shape)
    # all text includes the prefix
    all_text_sentences = []
    # prefixed_text_sentences excludes the prefix
    prefixed_text_sentences = []
    for idx in range(all_sentences.shape[0]):  # iterate over the batch dimension
        # sentence_id = sentence[0]
        idx_offset = 1 if args.pretrained_class=="ctrl" else 0
        prefixed_sentence = all_sentences[idx, idx_offset:].tolist()
        idx_offset += args.prefix_length
        sentence = all_sentences[idx, idx_offset:].tolist()
        
        decoded_sentence = tokenizer.decode(sentence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        prefixed_decoded_sentence = tokenizer.decode(prefixed_sentence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for idx in range(len(decoded_sentence))[::-1]:
            if decoded_sentence[idx]!="!":
                break

        decoded_sentence = decoded_sentence[:idx+1]
                
        for idx in range(len(prefixed_decoded_sentence))[::-1]:
            if prefixed_decoded_sentence[idx]!="!":
                break

        prefixed_decoded_sentence = prefixed_decoded_sentence[:idx+1]

        # all_text is without the prefix, prefixed is including the prefix.
        all_text_sentences.append(decoded_sentence)
        prefixed_text_sentences.append(prefixed_decoded_sentence)
    
    ###################################################################################
    ############################ Score the Generated Texts ############################ 
    ###################################################################################
    results_file = os.path.join("results/", args.pretrained_class, args.results_file)
    results_basename = os.path.basename(results_file).replace(".json", "")
    results_dir = os.path.dirname(results_file)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #results = {}  # moved to k/p/ent_prime
    scores = {}
    files = os.path.join("saved_generations/", results_basename, args.pretrained_class, args.output_dir, output_file)
    files_dir = os.path.dirname(files)
    if not os.path.isdir(files_dir):
        os.makedirs(files_dir)

    print(f"Writing generated sentences to {files}.")
    utils.write_sentences(all_text_sentences, files)

    preprocessed_files = os.path.join("preprocessed_generations/", results_basename, args.pretrained_class, args.output_dir, output_file)
    preprocessed_files_dir = os.path.dirname(preprocessed_files)
    if not os.path.isdir(preprocessed_files_dir):
        os.makedirs(preprocessed_files_dir)

    print(f"Writing preprocessed sentences to {preprocessed_files}.")
    preprocessed_sentences, filtered_indicies, filtered_lengths = utils.preprocess_text(prefixed_text_sentences, tokenizer, lmin=args.preprocessed_min, lmax=args.preprocessed_max)
    utils.write_sentences(preprocessed_sentences, preprocessed_files)

    # update the reference file to be chunked to our size
    reference_file = args.eval_text
    chunked_reference_file = f"{reference_file}_seq:{args.max_seq_length}_min:{args.preprocessed_min}_max:{args.preprocessed_max}_prefix:{args.prefix_length}_model:{args.pretrained_class.replace('models/', '')}"
    if not os.path.exists(chunked_reference_file):
        utils.lock(chunked_reference_file)
        print("Reference lock acquired!")
        # begin critical section!
        utils.chunk_and_prefix_file(reference_file, tokenizer, args.preprocessed_min, args.preprocessed_max, chunked_reference_file, prefix_length=args.prefix_length)
        # end critical section!
        utils.unlock(chunked_reference_file)

    filtered_tokenizations = []
    filtered_logprobs = []
    for idx in filtered_indicies: 
        filtered_tokenizations.append(all_sentences[idx])
        filtered_logprobs.append(all_logprobs[idx])
    filtered_tokenizations = torch.stack(filtered_tokenizations, dim=0)
    filtered_logprobs = torch.stack(filtered_logprobs, dim=0)

    del all_logprobs
    gc.collect()
    
    if args.eval_method=="BLEU":
        # use BLEU calculation
        smoothing_method = {"nist": SmoothingFunction().method3}
        for name, method in smoothing_method.items():
            scores[name] = utils.evaluate_bleu(files, chunked_reference_file, num_real_sentences=args.num_sentences, num_generated_sentences=args.num_sentences, gram=args.gram, smoothing_method=method, chunk_size=15)
            print()

        for name in smoothing_method.keys():
            results[name] = {}
            results[name]['scores'] = scores[name]

        results['nist']['scores']['bleu5'] = results['nist']['scores']['bleu5'] * -1.0
        bleu = results['nist']['scores']['bleu5'] * -1.0
        sbleu = results['nist']['scores']['self-bleu5']
    else:
        raise Exception("We don't support other automatic metrics!")    

    print("Results:", bleu, sbleu)
        
    ###################################################################################
    ############################# Result Reporting Section ############################ 
    ###################################################################################

    if not args.dry_run:
        results_file = os.path.join("results/", args.pretrained_class, args.results_file)
        results_dir = os.path.dirname(results_file)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        utils.lock(results_file)
        print("Lock acquired!")

        # begin critical section!
        if os.path.exists(results_file):
            with open(results_file, "r+") as f:
                current = json.load(f)
        else:
            current = {}

        key = output_file[:-4]
        current[key] = results 
        random_file = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))
        random_file = os.path.join("results/", args.pretrained_class, random_file)
        with open(random_file, "w+") as f:
            json.dump(current, f)

        os.rename(random_file, results_file)
        
        # save generations
        saved_tokens_file = os.path.join("tokens/", results_basename, args.pretrained_class, args.output_dir, output_file)
        saved_tokens_dir = os.path.dirname(saved_tokens_file)
        if not os.path.isdir(saved_tokens_dir):
            os.makedirs(saved_tokens_dir)

        saved_tokens = {}
        saved_tokens['args'] = [vars(args), vars(subparsers[args.sampler].parse_known_args()[0])]
        idx_offset = 1 if args.pretrained_class=="ctrl" else 0
        saved_tokens['with_prefix'] = all_sentences[:, idx_offset:].tolist()
        idx_offset += args.prefix_length
        saved_tokens['without_prefix'] = all_sentences[:, idx_offset:].tolist()

        with open("saved_tokens_file", "w+") as f:
            json.dump(saved_tokens, f)

        # save log probabilities
        preprocessed_logits = os.path.join("preprocessed_logprobs/", results_basename, args.pretrained_class, args.output_dir, output_file)
        preprocessed_logits_dir = os.path.dirname(preprocessed_logits)
        if not os.path.isdir(preprocessed_logits_dir):
            os.makedirs(preprocessed_logits_dir)

        d = {}
        print(filtered_logprobs.shape)
        for idx in range(filtered_logprobs.shape[0]):
            if preprocessed_sentences[idx] in d:
                raise Exception("Duplicate sentences found!")
            sent_id = hashlib.sha256(preprocessed_sentences[idx].encode()).hexdigest()
            d[sent_id] = {"model_score": filtered_logprobs[idx].item(), "lengths": filtered_lengths[idx] - args.prefix_length, "sentence": preprocessed_sentences[idx]} 
        print("Avg log probabilities:", (filtered_logprobs / (torch.tensor(filtered_lengths) - args.prefix_length)).mean(dim=0))

        with open(preprocessed_logits, "w") as f:
            json.dump(d, f)

        # create plot
        plots_file = os.path.join("plots/", args.pretrained_class, args.results_file)
        plots_dir = os.path.dirname(plots_file)
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)

        plot = plotter.Plotter(results_file)
        plot.plot_curves()
        if args.plot_gold:
            params = {"eval_method": args.eval_method, "chunk": args.max_seq_length, "ngram": args.gram, "knn": args.knn,  "num_sentences": args.num_sentences}
            result = plot.plot_gold(params)
            if not result:
                # We don't have a proper score for our reference file, so let's go ahead and create it.
                params['gold_file'] = chunked_reference_file.replace("test", "valid") 
                print(f"Evaluating gold point on {params['gold_file']} with KNN={args.knn}")
                params['num_sentences'] = args.num_sentences
                params['reference_corpus'] = chunked_reference_file
                params['chunk'] = args.max_seq_length
                params['eval_method'] = args.eval_method
                params['knn'] = args.knn
                params['gram'] = args.gram
                params['device'] = device
                score_gold(params) 
                result = plot.plot_gold(params)

        plot.save(plots_file.replace(".json", ""))
        # end critical section!
        utils.unlock(results_file)

if __name__=="__main__":
    args, subparsers = parse_args()
    main(args, subparsers)
