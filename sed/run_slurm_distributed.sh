#!/bin/bash

#SBATCH -A EUHPC_D11_069
#SBATCH -p boost_usr_prod
#SBATCH -o /leonardo_scratch/fast/EUHPC_D11_069/repos/SED/output/train_slurm_out.txt
#SBATCH -e /leonardo_scratch/fast/EUHPC_D11_069/repos/SED/output/train_slurm_error.txt
#SBATCH --time 20:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2        # 4 gpus per node out of 4
#SBATCH --mem=70000         # memory per node out of 494000MB (481GB)
#SBATCH --job-name=train

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

module load profile/deeplrn
module load cineca-ai/4.3.0

cd /leonardo_scratch/fast/EUHPC_D11_069/repos/
source ./.sed/bin/activate
cd /leonardo_scratch/fast/EUHPC_D11_069/repos/SED/

export TF_ENABLE_ONEDNN_OPTS=0

python train_net_distributed.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --resume \
 OUTPUT_DIR $output \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS models/model_final.pth \
 $opts

exit # ends the salloc allocation