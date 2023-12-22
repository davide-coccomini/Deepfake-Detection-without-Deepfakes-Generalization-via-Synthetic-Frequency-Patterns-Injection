#!/bin/bash
#SBATCH --N 1 # number of nodes to allocate
#SBATCH --ntasks-per-node=1 # number of processes per compute node
#SBATCH --cpus-per-task=32 # number of cores allocated per process
#SBATCH --gres=gpu:4 # number of gpus per node
#SBATCH --t 24:00:00 # total run time of the job allocation
#SBATCH --mem=0 # memory request per node
#SBATCH --o generation_out.out  # for stdout redirection
#SBATCH --e generation_error.err  # for stderr redirection
#SBATCH --p boost_usr_prod # partition for resource allocation
#SBATCH --A IscrC_DIFD  # account name for allocation
module load python/3.10.8--gcc--11.3.0 
#module load openmpi
module load profile/deeplrn
module load cineca-ai

export OMP_PROC_BIND=true
export MASTER_ADDR=$(hostname)
#export OMP_NUM_THREADS=32
#export MKL_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_PORT=29500  


mpirun accelerate launch  --num_processes=4 generate_images.py  > output.out