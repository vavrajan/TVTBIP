#!/bin/bash
#SBATCH --job-name=TVTBIP-simulation # short name for your job
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=350000 # in megabytes, default is 4GB per task
#SBATCH -o /home/jvavra/TVTBIP/out/simulation/%x_%j_%N.out      # save stdout to file. The filename is defined through filename pattern
#SBATCH -e /home/jvavra/TVTBIP/err/simulation/%x_%j_%N.err      # save stderr to file. The filename is defined through filename pattern
#SBATCH --time=11:59:59 # total run time limit (HH:MM:SS)

. /opt/apps/2023-04-11_lmod.bash
ml purge
ml miniconda3-4.10.3-gcc-12.2.0-ibprkvn

cd /home/jvavra/TVTBIP/
conda activate tf_TBIP

python3 /home/jvavra/TVTBIP/code/simulation/explore_sampled_counts.py  --seed=314159  --data_name=hein-daily  --simulation=simulation
