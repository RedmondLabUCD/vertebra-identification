#!/bin/bash
#SBATCH --ntasks=1           ### How many CPU cores do you need?
#SBATCH --mem=20G            ### How much RAM memory do you need?
#SBATCH -p long          ### The queue to submit to: express, short, long, interactive
#SBATCH --gres=gpu:1         ### How many GPUs do you need?
#SBATCH -t 6-00:00:00        ### The time limit in D-hh:mm:ss format
#SBATCH -o /trinity/home/r094879/repositories/vertebra-identification/output/out3_%j.log       ### Where to store the console output (%j is the job number)
#SBATCH -e /trinity/home/r094879/repositories/vertebra-identification/error/error3_%j.log      ### Where to store the error output
#SBATCH --job-name=sp_lm_model  ### Name your job so you can distinguish between jobs

# ----- Load the modules -----
module purge
module load Python/3.9.5-GCCcore-10.3.0

# If you need to read/write many files quickly in tmp directory use:
source "/tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env"

# ----- Activate virtual environment -----
# Do this after loading python module
source /trinity/home/r094879/vertebra-detection/bin/activate

# ----- Your tasks -----
# python final_training.py UNet_LM_CL3 --custom_loss True
python final_training.py UNet_deep_CL3 --custom_loss True
