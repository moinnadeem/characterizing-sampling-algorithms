#!/bin/bash
# ==========================================================================================
# ===================================== PARAMETER SWEEPS ===================================== 
# ==========================================================================================

P_BASE_VARS=(0.10 0.20 0.30 0.35 0.40 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
K_BASE_VARS=(10 30 50 100 250 500 1500 3000 8000) 
KTEMP_VARS=(0.9 0.8 0.7 0.6)

LAGS=(1 2 3 4)
LINEAR_BASES=(2 3 4 5)

#K_RANDOMNESS_VARS=(1 2 5 10 20 30 40 50 60 80 90 100 150 200 300 500 1000 2000 5000 10000)
K_RANDOMNESS_VARS=(1 2 5 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 300 500 1000 2000 5000 10000)
P_RANDOMNESS_VARS=(0.01 0.02 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 0.95)
P_RANDOMNESS_BASES=(0.01 0.02 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8) 

TEMPERATURE_VARS=(0.01 0.05 0.1 0.3 0.5 0.6 0.7 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.20 1.25 1.30 1.35 1.40 1.50 1.60 1.70 1.8 1.9 2.0 3.0 4.0 5.0 8.0 10.0 15.0 20.0 30.0 50.0 100.0)

K_TRUNCATION_VALUES=(0 1 2 5 10 20)
KNN_RANGES=(2 3 4 5 6 10 20 30 40 50 70 90 100 150 200)

# ==========================================================================================
# ===================================== CONFIGURATION ===================================== 
# ==========================================================================================

N_GPUS=30

PRETRAINED_CLASS="models/gigaword_gpt2"

GENERATION_BATCH_SIZE=80
ENCODING_BATCH_SIZE=1

PREFIX_FILE="data/gigaword/test.txt_filtered"
PREFIX_LENGTH=10
GRAM=3
CTRL_CODE="Wikipedia"
MAX_SEQ_LENGTH=100
NUM_SENTENCES=6000
SEED=10
EVAL_TEXT="data/gigaword/valid.txt_filtered"
RESULTS_FILE="results_gigaword_amt_refreshed_num$NUM_SENTENCES.json"
EVAL_METHOD="BLEU"
KNN=7
PLOT_GOLD=0

export PRETRAINED_CLASS BATCH_SIZE PREFIX_FILE GRAM CTRL_CODE KNN PLOT_GOLD 
export MAX_SEQ_LENGTH NUM_SENTENCES SEED EVAL_TEXT RESULTS_FILE EVAL_METHOD PREFIX_LENGTH
export GENERATION_BATCH_SIZE ENCODING_BATCH_SIZE
GPU_IDX=0

mkdir -p slurm_output

# ==========================================================================================
# ======================================= KNN SWEEP ======================================== 
# ==========================================================================================

#for KNN in ${KNN_RANGES[@]} ; do 
  #RESULTS_FILE="knn/results_seed_20_8k_sentences_embedding_sanity_${KNN}.json"
  #export RESULTS_FILE KNN
  #IS_TOP_P=1
  #export IS_TOP_P
  #for BASE in ${P_BASE_VARS[@]} ; do
    #export BASE 
    #sbatch -J sweep_$GPU_IDX -d singleton run_job.sh FixedScheduler
    #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
  #done
#done

# ==========================================================================================
# ===================================== TOP-K SAMPLING ===================================== 
# ==========================================================================================

for BASE in ${K_BASE_VARS[@]} ; do
    export BASE 
    sbatch -J sweeps run_job.sh FixedScheduler
    GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
done
unset BASE

#for NEGATIVE_BASE in ${K_TRUNCATION_VALUES[@]} ; do
  #for BASE in ${K_BASE_VARS[@]} ; do
    #export BASE NEGATIVE_BASE 
    #sbatch -J sweep_$GPU_IDX -d singleton run_job.sh NegativeScheduler
    #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
  #done
#done

#for BASE in ${LINEAR_BASES[@]} ; do
    #export BASE 
    #sbatch -J sweep_$GPU_IDX -d singleton run_job.sh LinearScheduler
    #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
#done

#for BASE in ${LINEAR_BASES[@]} ; do
  #for LAG in ${LINEAR_BASES[@]} ; do
    #export BASE LAG
    #sbatch -J sweep_$GPU_IDX -d singleton run_job.sh LinearLagScheduler
    #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
  #done
#done

#BASE=1
#export BASE
#for MAX_RANDOMNESS in ${K_RANDOMNESS_VARS[@]} ; do
  #export BASE MAX_RANDOMNESS 
  #sbatch -J sweep_$GPU_IDX -d singleton run_job.sh RandomFixedScheduler 
  #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
#done
#unset BASE MAX_RANDOMNESS

# ==========================================================================================
# ================================== TEMPERATURE SAMPLING ================================== 
# ==========================================================================================

#BASE=1.0
#IS_TOP_P=1
#export BASE IS_TOP_P

#for TEMPERATURE in ${TEMPERATURE_VARS[@]} ; do
	#export TEMPERATURE 
	#sbatch -J sweep_$GPU_IDX -d singleton run_job.sh TemperatureSweep 
	#GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
#done
#unset BASE IS_TOP_P TEMPERATURE

#for BASE in ${K_BASE_VARS[@]} ; do
  #for TEMPERATURE in ${KTEMP_VARS[@]} ; do
    #export TEMPERATURE BASE 
    #sbatch -J sweep_$GPU_IDX -d singleton run_job.sh KTemperatureSweep 
    #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
  #done
#done

# ==========================================================================================
# ===================================== TOP-P SAMPLING ===================================== 
# ==========================================================================================

#IS_TOP_P=1
#export IS_TOP_P
#for BASE in ${P_BASE_VARS[@]} ; do
	#export BASE 
	#sbatch -J sweep_$GPU_IDX -d singleton run_job.sh FixedScheduler
	#GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
#done

#BASE=0
#for BASE in ${P_RANDOMNESS_BASES[@]} ; do
  #for MAX_RANDOMNESS in ${P_RANDOMNESS_VARS[@]} ; do
    #export BASE MAX_RANDOMNESS 
    #sbatch -J sweep_$GPU_IDX -d singleton run_job.sh RandomFixedScheduler 
    #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
  #done
#done

#unset BASE IS_TOP_P MAX_RANDOMNESS

# ==========================================================================================
# ===================================== JOINT SAMPLING ===================================== 
# ==========================================================================================

#for TOP_K in ${K_BASE_VARS[@]} ; do
  #for TOP_P in ${P_BASE_VARS[@]} ; do
    #export TOP_P TOP_K 
    #sbatch -J sweep_$GPU_IDX run_job.sh JointScheduler 
    #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
  #done
#done

#unset TOP_K
#unset TOP_P

#echo ==========================================================================================
#echo ============================= RANDOM TOP-K SAMPLING ====================================== 
#echo ==========================================================================================
#sleep 2
 
#unset IS_TOP_P
#BASE=1
#export BASE
#for MAX_RANDOMNESS in ${K_RANDOMNESS_VARS[@]} ; do
  #export BASE MAX_RANDOMNESS 
  #echo BASE $BASE MAX_RANDOMNESS $MAX_RANDOMNESS RandomFixedScheduler 
  #sbatch -J R${MAX_RANDOMNESS}K${BASE} -d singleton run_job.sh RandomFixedScheduler
  #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
#done
#unset BASE MAX_RANDOMNESS

#if [ 1 -eq 1 ]; then
  #echo ==========================================================================================
  #echo ============================ Sorted Noised SAMPLING ====================================== 
  #echo ==========================================================================================
  #sleep 2
  #for BASE in 2 3 5 7 10 20 30 50 60 70 80 100 200 300 400 500 600 800 1000 2000 5000 10000 ; do
    #for NOISE_WEIGHT in 0.001 0.002 0.003 0.005 ; do
      #export BASE NOISE_WEIGHT
      #echo BASE $BASE NOISE_WEIGHT $NOISE_WEIGHT SortedNoisedFixedScheduler 
      #sbatch -J NW${NOISE_WEIGHT}B${BASE} -d singleton run_job.sh SortedNoisedFixedScheduler
      #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
    #done
  #done
#fi

#echo ==========================================================================================
#echo =========================== MAX ENTROPY SAMPLING ====================================== 
#echo ==========================================================================================
#sleep 2
#for BASE in 0.01 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.5 2.75 3 3.25 3.5 3.75 4 4.5; do
  #export BASE 
  #echo BASE $BASE MaxEntropyScheduler
    #sbatch -J sweep_$GPU_IDX -d singleton run_job.sh MaxEntropyScheduler
  #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
#done

#if [ 1 -eq 1 ]; then
  #echo ==========================================================================================
  #echo ======================== RandomSpace Topk SAMPLING ====================================== 
  #echo ==========================================================================================
  #PRESERVE_LARGEST_PROB=0
  #export PRESERVE_LARGEST_PROB
  #echo PRESERVE_LARGEST_PROB $PRESERVE_LARGEST_PROB
  #sleep 2 
  #for BASE in 50257 ; do
    #for RANDOMSPACE_RATE in 0.001 0.01 0.02 0.05 0.10 0.20 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.97 0.99 ; do
      #export BASE RANDOMSPACE_RATE
      #echo BASE $BASE SPACE $RANDOMSPACE_RATE RandomSpaceTopkScheduler 
      #sbatch -J B${BASE}R${RANDOMSPACE_RATE} -d singleton run_job.sh RandomSpaceTopkScheduler
      #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
    #done
  #done
#fi

#if [ 1 -eq 1 ]; then
  #echo ==========================================================================================
  #echo =========================== TARGET ENTROPY SAMPLING ====================================== 
  #echo ==========================================================================================
  #sleep 2
  #for BASE in 0.01 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.5 2.75 3 3.25 3.5 3.75 4 4.5; do
    #export BASE 
    #echo BASE $BASE TargetEntropyScheduler
    #sbatch -J sweep_$GPU_IDX -d singleton run_job.sh TargetEntropyScheduler
    #GPU_IDX=$(((GPU_IDX+1) % N_GPUS))
  #done
#fi
