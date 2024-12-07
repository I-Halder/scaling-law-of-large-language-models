#!/bin/bash
#
#SBATCH --job-name=aSFT
#SBATCH --out="aSFT-%A_%a.out"
#SBATCH --cpus-per-task=32
#SBATCH --mem=72G
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab

echo $(hostname)

module load python/3.10.13-fasrc01
module load Anaconda2/2019.10-fasrc01  
conda activate /n/netscratch/pehlevan_lab/Everyone/indranilhalder/env/env_LLM

python aSFT.py