#!/bin/bash
#SBATCH --partition=gpu,normal # use 1080 partition
# comment change #--dependency=singleton
#SBATCH --gres=gpu:1           # grab 2 gpus
#SBATCH --ntasks=1             # going to run 2 tasks
#SBATCH --nodes=1-1            # restrict job to a single node (min of 1, max of 1)
#SBATCH --time=4:00:00         # set a time limit (e.g. 4 hours) for the job before it times-out (good practice!)
#SBATCH --mem=50G
#SBATCH --output=slurm_output/R.%j.out
# print parameters

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_TASKID="$SLURM_ARRAY_TASK_ID
# now define each task to run=

rm -r slurm_output
mkdir slurm_output

srun -n1 --gres=gpu:1 python3 main.py $1
