from collections import defaultdict
import random
import argparse
import json
import os
import utils
from nltk.translate.bleu_score import SmoothingFunction
import embeddings
import torch

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--gold-file", type=str, default=os.environ.get("GOLD_FILE", ""))
    parser.add_argument("--num-gold-sentences", type=int, default=os.environ.get("NUM_GOLD_SENTENCE", ""))
    parser.add_argument("--num-ref-sentences", type=int, default=os.environ.get("NUM_REF_SENTENCES", ""))
    parser.add_argument("--reference-corpus", type=str, default=os.environ.get("REFERENCE_CORPUS", ""))
    parser.add_argument("--chunk", type=int, default=os.environ.get("CHUNK", 10))
    parser.add_argument("--gram", default=os.environ.get("GRAM",3), type=int)
    parser.add_argument("--eval-method", choices=["BLEU", "Embedding"], default=os.environ.get("EVAL_METHOD","BLEU"))
    parser.add_argument("--encoding-batch_size", default=os.environ.get("ENCODING_BATCH_SIZE", 500))
    parser.add_argument("--knn", type=int, default=os.environ.get("KNN", 10))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"), choices=["cpu", "gpu"])
    parser.add_argument("--seed", default=os.environ.get("SEED", 5), type=int)
    return parser.parse_args() 

def score_gold(params): 
    utils.print_args(args)
    utils.set_seed(args.seed)

    gold_file = params['gold_file']
    gold_filename = os.path.basename(gold_file)
    with open(gold_file, "r") as f:
        print("Gold file length:", len(f.readlines()))
    reference_file = params['reference_corpus']
    with open(reference_file, "r") as f:
        print("Reference file length:", len(f.readlines()))
    chunk = params['chunk']
    ngram = params['gram']
    num_ref_sentences = params['num_ref_sentences']
    num_gold_sentences = params['num_gold_sentences']
    evaluation_method = params['eval_method']
    device = params['device']
    output_file = "results/gold_corpora/results.txt"

    results = {}
    if evaluation_method=="BLEU":
        gram = params['gram']

        smoothing_method = {"nist": SmoothingFunction().method3}
        scores = {}

        for name, method in smoothing_method.items():
            scores[name] = utils.evaluate_bleu(gold_file, reference_file, num_real_sentences=num_ref_sentences, 
                    num_generated_sentences=num_gold_sentences, gram=gram, smoothing_method=method, chunk_size=15)

        print(scores)
        scores['nist']['scores'] = {}
        scores['nist']['scores']['bleu5'] = scores['nist']['bleu5'] * -1.0
        for name in smoothing_method.keys():
            results[name] = {}
            results[name]['scores'] = scores[name]
            results['num_ref_sentences'] = num_ref_sentences 
            results['num_gold_sentences'] = num_gold_sentences 
    elif evaluation_method=="Embedding":
        knn = params['knn']

        # use Embedding calculation
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')
        # TODO: update chunks to begin more naturally.
        all_text_sentences = []
       	with open(reference_file, "r+") as f:
            reference_sentences = [i.replace("</s>", "\n") for i in f.readlines()] 
        with open(gold_file, "r+") as f:
            all_text_sentences = [i.replace("</s>", "\n") for i in f.readlines()] 	 
        encoding_batch_size = 500 
        candidate_sentences = all_text_sentences 

        random.shuffle(candidate_sentences)
        candidate_sentences = candidate_sentences[:num_sentences]

        random.shuffle(reference_sentences)
        reference_sentences = reference_sentences[:num_sentences]
        # TODO: split the embeddings up into sentences
        # and mean-pool the sentences to get better embeddings.
        print("Beginning embedding eval...")
        print("Lengths:", len(candidate_sentences), len(reference_sentences))
        encoding_batch_size = 500 
        
        reference_embeddings = embeddings.encode_sentences(sentence_model, reference_sentences, batch_size=encoding_batch_size, device=device)
        candidate_embeddings = embeddings.encode_sentences(sentence_model, candidate_sentences, batch_size=encoding_batch_size, device=device)

        bleu = embeddings.compute_scores(candidate_sentences, candidate_embeddings, 
                reference_sentences, reference_embeddings, k=knn).item() * -1.0

        sbleu = embeddings.compute_scores(candidate_sentences, candidate_embeddings, 
                candidate_sentences, candidate_embeddings, k=knn, is_self=True).item() 

        results['knn'] = knn
        results['chunk'] = chunk
        results['num_sentences'] = num_sentences
        results['nist'] = {}
        results['nist']['scores'] = {}
        results['nist']['scores']['bleu5'] = bleu
        results['nist']['scores']['self-bleu5'] = sbleu


    if os.path.exists(output_file):
        with open(output_file, "r+") as f:
            current = json.load(f)
    else:
        current = {"BLEU": defaultdict(lambda: defaultdict({})),
                "Embedding": defaultdict(lambda: defaultdict({}))} 

    if evaluation_method=="BLEU":
        num_sentences = f"{num_ref_sentences}-{num_gold_sentences}"
        if num_sentences not in current['BLEU']: 
            current['BLEU'][num_sentences] = {}
        if chunk not in current['BLEU'][num_sentences]: 
            current['BLEU'][num_sentences][chunk] = {} 
        if ngram not in current['BLEU'][num_sentences][chunk]: 
            current['BLEU'][num_sentences][chunk] = {} 

        current['BLEU'][num_sentences][chunk][ngram] = results

    else:
        if knn not in current['Embedding']: 
            current['Embedding'][knn] = {}
        if chunk not in current['Embedding'][knn]: 
            current['Embedding'][knn][chunk] = {} 
        if num_sentences not in current['Embedding'][knn][chunk]:
            current['Embedding'][knn][chunk][num_sentences] = {}

        current['Embedding'][knn][chunk][num_sentences] = results

    with open(output_file, "w+") as f:
        json.dump(current, f)

if __name__=="__main__":
    args = parse_args()
    params = vars(args) 
    score_gold(params)  
