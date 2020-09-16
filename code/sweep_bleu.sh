#!/bin/bash
# ==========================================================================================
# ===================================== PARAMETER SWEEPS ===================================== 
# ==========================================================================================

NUM_GOLD_SENTENCES=(4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000) 

# ==========================================================================================
# ===================================== CONFIGURATION ===================================== 
# ==========================================================================================

GOLD_FILE="data/gigaword/test.txt_filtered_seq:100_min:40_max:50_prefix:10_model:gpt2"
NUM_GOLD_SENTENCE=10000
NUM_REF_SENTENCES=10000
REFERENCE_CORPUS="data/gigaword/valid.txt_filtered_seq:100_min:40_max:50_prefix:10_model:gpt2"
CHUNK=10
GRAM=3
EVAL_METHOD="BLEU"
ENCODING_BATCH_SIZE=500
KNN=10
DEVICE="cpu"
SEED=5
export SEED DEVICE KNN ENCODING_BATCH_SIZE EVAL_METHOD GRAM CHUNK REFERENCE_CORPUS NUM_REF_SENTENCES NUM_GOLD_SENTENCE GOLD_FILE

# ==========================================================================================
# ========================================= SWEEP ========================================== 
# ==========================================================================================

for NUM_GOLD_SENTENCE in ${NUM_GOLD_SENTENCES[@]} ; do
    export NUM_GOLD_SENTENCE 
    echo "This is the shuffled version!"
    sbatch -J bleu run_bleu.sh 
done
