#!/bin/bash
# ==========================================================================================
# ===================================== CONFIGURATION ===================================== 
# ==========================================================================================

N_GPUS=400

PRETRAINED_CLASS="models/wikitext_gpt2"

GENERATION_BATCH_SIZE=30
ENCODING_BATCH_SIZE=1

PREFIX_FILE="data/wikitext/modified_wiki.test.raw"
PREFIX_LENGTH=10
GRAM=3
CTRL_CODE="Wikipedia"
MAX_SEQ_LENGTH=100
NUM_SENTENCES=9000
SEED=10
EVAL_TEXT="data/wikitext/modified_wiki.valid.raw"
RESULTS_FILE="results_gigaword_amt_5_num$NUM_SENTENCES.json"
EVAL_METHOD="BLEU"
KNN=7
PLOT_GOLD=0

export PRETRAINED_CLASS BATCH_SIZE PREFIX_FILE GRAM CTRL_CODE KNN PLOT_GOLD 
export MAX_SEQ_LENGTH NUM_SENTENCES SEED EVAL_TEXT RESULTS_FILE EVAL_METHOD PREFIX_LENGTH
export GENERATION_BATCH_SIZE ENCODING_BATCH_SIZE
GPU_IDX=0

# ==========================================================================================
# =========================================== JOB ========================================== 
# ==========================================================================================

BASE=100
export BASE 
sbatch -J sweep_$GPU_IDX -d singleton run_job.sh FixedScheduler
