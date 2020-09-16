#!/bin/bash
#SBATCH --partition=gpu # use 1080 partition
#SBATCH --gres=gpu:1           # grab 2 gpus
#SBATCH --ntasks=1             # going to run 2 tasks
#SBATCH --nodes=1-1            # restrict job to a single node (min of 1, max of 1)
#SBATCH --time=4:00:00         # set a time limit (e.g. 4 hours) for the job before it times-out (good practice!)
#SBATCH --output=output/R-%x.%j.out
# SBATCH -t 1-1000%30

echo "11745@MIT" | renew

P_BASE_VARS=(0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
K_BASE_VARS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
TANH_WINDOW=(20 25 30 35 40 45 50 55 60)
TANH_BASE=(5 6 7 10 13 15 17)
LAGS=(1 2 3 4)
LINEAR_BASES=(2 3 4 5)
MAX_SEQ_LENGTH=50
NUM_SENTENCES=8000
rm -r output 
mkdir output 

rm -r saved_generations
mkdir saved_generations 

rm -r plots
mkdir plots

export NEGATIVE_K MAX_SEQ_LENGTH NUM_SENTENCES RESULTS_FILE

for BASE in ${K_BASE_VARS[@]} ; do
    export BASE 
    srun python3 main.py FixedScheduler
done

for BASE in ${TANH_BASE[@]} ; do
  for WINDOW in ${TANH_WINDOW[@]}; do
    export BASE WINDOW 
    srun python3 main.py TanhScheduler 
  done
done

for BASE in ${K_BASE_VARS[@]} ; do
    export BASE 
    srun python3 main.py LinearScheduler
done

for BASE in ${LINEAR_BASES[@]} ; do
    export BASE 
    srun python3 main.py LinearScheduler
done

for BASE in ${LINEAR_BASES[@]} ; do
  for LAG in ${LINEAR_BASES[@]} ; do
    export BASE LAG
    srun python3 main.py LinearLagScheduler
  done
done

unset WINDOW

IS_TOP_P=1
export IS_TOP_P
for BASE in ${P_BASE_VARS[@]} ; do
  export BASE 
  srun python3 main.py FixedScheduler
done
