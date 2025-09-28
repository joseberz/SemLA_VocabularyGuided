#!/bin/bash

export DETECTRON2_DATASETS=/leonardo_work/EUHPC_D11_069/datasets/
export PYTHONUNBUFFERED=0

module load profile/deeplrn
module load cineca-ai/4.3.0

cd /leonardo_scratch/fast/EUHPC_D11_069/repos/
source ./.sed/bin/activate
cd /leonardo_scratch/fast/EUHPC_D11_069/repos/SED/

export TF_ENABLE_ONEDNN_OPTS=0

unbuffer python experiments_GL.py "$1" "$2"
