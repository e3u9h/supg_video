#!/bin/bash
#SBATCH --job-name=TEST
#SBATCH --mail-user=fzdu2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d2/spc/fzdu2/supg/supg/experiments/output7.txt #Do not use "~" point to your home!
#SBATCH --gres=gpu:1

REM python3 newexp.py --source newout.mp4 --text 'a panda' --save  --oth 0.6
python3 label.py
