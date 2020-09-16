import transformers
import utils
from pprint import pprint
from nltk.tokenize import word_tokenize
from tqdm import tqdm

pretrained_class = "gpt2"
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_class)
reference_file = "data/modified_wiki.valid.raw" 

backtrace = {}
for max_seq_length in tqdm(range(20, 150, 5)):
    chunked_reference_file = f"{reference_file}_{max_seq_length}_{pretrained_class}"
    utils.chunk_file(reference_file, tokenizer, max_seq_length, chunked_reference_file)
    
    # consume file
    with open(chunked_reference_file, "r") as f:
        # note that each line is a sample
        lines = f.readlines()

    lens = []
    for line in lines:
        # figure out location of last EOS [. ! ?]
        p = line.rfind(".")
        c = line.rfind("?")
        e = line.rfind("!")
        end = max([p, c, e])
        if end==-1:
            continue
            end=len(line)

        # take that substring and tokenize it
        tokens = word_tokenize(line[end:])
        lens.append(len(tokens))

    avg_left_tokens = sum(lens) / len(lens)
    avg_size = len(lines) / len(lens)

    # preserve number of words and the seq length 
    backtrace[max_seq_length] = (avg_left_tokens, avg_size)

pprint(backtrace)
