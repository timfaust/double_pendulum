#!/bin/bash
#
#SBATCH --job-name=erfan-rl
#
# Where to send job start / end messages to - comment in to use!
#SBATCH --mail-user=erfan.aghadavoodi@gmail.com
#SBATCH --mail-type=ALL
#
# How many cpus to request
#SBATCH -c 1
#
# How much RAM to request
#SBATCH --mem-per-cpu=8G
#
# How many gpus to request
#SBATCH --gres=gpu:1
#
# Limit runtime d-hh:mm:ss - here limited to 1min
#SBATCH -t 00:40:00
#
# Submit a stud job
#SBATCH -p stud
#
#
# Define standard output files - make sure those files exist
#SBATCH --output=/mnt/beegfs/home/stud_aghadavoodi/double_pendulum/examples/reinforcement_learning/General/sbatch.output
#SBATCH --error=/mnt/beegfs/home/stud_aghadavoodi/double_pendulum/examples/reinforcement_learning/General/sbatch.error
#
# Computation area begins here
#
echo "Loading Modules!"
srun cd /mnt/beegfs/home/stud_aghadavoodi/double_pendulum/examples/reinforcement_learning/General/ || exit 2

srun srun --pty bash conda init
srun conda activate double_pendulum

srun export PYTHONPATH="/mnt/beegfs/home/stud_aghadavoodi/double_pendulum/src/python:$PYTHONPATH"
srun export PYTHONPATH="/mnt/beegfs/home/stud_aghadavoodi/double_pendulum/:$PYTHONPATH"
echo "Starting Training on slurm!"
srun python main.py

exit 0
