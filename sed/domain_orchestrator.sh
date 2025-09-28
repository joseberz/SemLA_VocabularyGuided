#!/bin/bash

#SBATCH -A EUHPC_D11_069
#SBATCH -p boost_usr_prod
#SBATCH -o /leonardo_scratch/fast/EUHPC_D11_069/repos/LoraRetrieval/output/domain_orchestator_output.txt
#SBATCH -e /leonardo_scratch/fast/EUHPC_D11_069/repos/LoraRetrieval/output/domain_orchestator_error.txt
#SBATCH --time 00:10:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=30000         # memory per node out of 494000MB (481GB)
#SBATCH --job-name=merge_compare

module load profile/deeplrn
module load cineca-ai/4.3.0

cd /leonardo_scratch/fast/EUHPC_D11_069/repos/LoraRetrieval/
source ./.catseg/bin/activate

export TF_ENABLE_ONEDNN_OPTS=0

python domain_orchestrator.py