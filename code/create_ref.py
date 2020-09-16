import argparse
import transformers
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--max-seq-length", default=100, type=int)
    parser.add_argument("--min", default=40, type=int)
    parser.add_argument("--max", default=50, type=int)
    parser.add_argument("--prefix-length", default=10, type=int)
    return parser.parse_args()


def main(args):
    pretrained_class = "gpt2"
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_class)
    num_tokens = tokenizer.add_special_tokens({"additional_special_tokens": ["</s>"], "unk_token": "<unk>"})
    print(f"Added {num_tokens} tokens!")
    reference_file = args.reference_file 
    max_seq_length = args.max_seq_length 
    chunked_reference_file = f"{reference_file}_seq:{args.max_seq_length}_min:{args.min}_max:{args.max}_prefix:{args.prefix_length}_model:gpt2"
    utils.chunk_and_prefix_file(reference_file, tokenizer, args.min, args.max, chunked_reference_file, prefix_length=args.prefix_length)

if __name__=="__main__":
    args = parse_args()
    main(args)
