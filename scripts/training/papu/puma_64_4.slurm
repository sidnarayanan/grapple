#!/bin/bash
#SBATCH -J sluma_4GPUs
#SBATCH -o sluma_4GPUs_%j.out
#SBATCH -e sluma_4GPUs_%j.err
#SBATCH --mail-user=sidn@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --exclude=node0024

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=pumappc
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

cd /home/${USER}/puma/grapple/
export PYTHONPATH=${PYTHONPATH}:${PWD}
cd -
nvidia-smi


ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
echo " Run started at:- "
date

## Horovod execution
python /home/${USER}/puma/grapple/scripts/training/papu/train_pu.py -c /home/${USER}/puma/grapple/scripts/training/papu/64_4_100_8x64_optimized_lr.yaml

echo "Run completed at:- "
date
