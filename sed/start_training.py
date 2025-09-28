from itertools import combinations
import subprocess

import os
from pathlib import Path

# SEEDS = [1000, 2000, 3000, 4000, 5000]
SEEDS = [1000]


def get_sharded_names(root):
    try:
        # Create a Path object for the directory
        path = Path(root)
        # Use the glob method to list all sub-directories
        subdirectories = [subdir.name for subdir in path.iterdir() if subdir.is_dir()]
        return subdirectories
    except FileNotFoundError:
        print(f"The directory {root} does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied for accessing {root}.")
        return []
_root = os.getenv("DETECTRON2_DATASETS", "datasets")

commands_cs = [
    # *[f"sbatch run_slurm.sh configs/cs/multidomain_fft.yaml 1 ./output/cs/cs-multidomain_fullfinetune_index-{seed} MODEL.LORA.ENABLED False  MODEL.LORA.NAME cs-multidomain_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/cs/multidomain_lora.yaml 1 ./output/cs/cs-multidomain_index-{seed} MODEL.LORA.NAME cs-multidomain_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    *[f"sbatch run_slurm.sh configs/cityscapes/normal/lora-normal.yaml 1 ./output/cs/cs-normal_index-{seed} MODEL.LORA.NAME cs-normal_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/cityscapes/normal/lora-normal.yaml 1 ./output/cs/cs-normal_fullfinetune_index-{seed} MODEL.LORA.NAME cs-normal_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/cs/rain/25mm/lora-rain-25mm.yaml 1 ./output/cs/cs-rain-25mm_index-{seed} MODEL.LORA.NAME cs-rain-25mm_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/cs/rain/50mm/lora-rain-50mm.yaml 1 ./output/cs/cs-rain-50mm_index-{seed} MODEL.LORA.NAME cs-rain-50mm_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/cs/rain/75mm/lora-rain-75mm.yaml 1 ./output/cs/cs-rain-75mm_index-{seed} MODEL.LORA.NAME cs-rain-75mm_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/cs/rain/100mm/lora-rain-100mm.yaml 1 ./output/cs/cs-rain-100mm_index-{seed} MODEL.LORA.NAME cs-rain-100mm_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/cs/rain/200mm/lora-rain-200mm.yaml 1 ./output/cs/cs-rain-200mm_index-{seed} MODEL.LORA.NAME cs-rain-200mm_index-{seed} SEED {seed}" for seed in SEEDS],
]

commands_acdc = [
    # *[f"sbatch run_slurm.sh configs/acdc/multidomain_fft.yaml 1 ./output/acdc/acdc-multidomain_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME acdc-multidomain_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/acdc/multidomain_lora.yaml 1 ./output/acdc/acdc-multidomain_index-{seed} MODEL.LORA.NAME acdc-multidomain_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/acdc/rain/lora-rain-acdc.yaml 1 ./output/acdc/acdc-rain_index-{seed} MODEL.LORA.NAME acdc-rain_index-{seed} SEED {seed}" for seed in SEEDS],
    *[f"sbatch run_slurm.sh configs/acdc/rain/lora-rain-acdc.yaml 1 ./output/acdc/acdc-rain_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME acdc-rain_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/acdc/fog/lora-fog-acdc.yaml 1 ./output/acdc/acdc-fog_index-{seed} MODEL.LORA.NAME acdc-fog_index-{seed} SEED {seed}" for seed in SEEDS],
    *[f"sbatch run_slurm.sh configs/acdc/fog/lora-fog-acdc.yaml 1 ./output/acdc/acdc-fog_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME acdc-fog_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/acdc/snow/lora-snow-acdc.yaml 1 ./output/acdc/acdc-snow_index-{seed} MODEL.LORA.NAME acdc-snow_index-{seed} SEED {seed}" for seed in SEEDS],
    *[f"sbatch run_slurm.sh configs/acdc/snow/lora-snow-acdc.yaml 1 ./output/acdc/acdc-snow_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME acdc-snow_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/acdc/night/lora-night-acdc.yaml 1 ./output/acdc/acdc-night_index-{seed} MODEL.LORA.NAME acdc-night_index-{seed} SEED {seed}" for seed in SEEDS],
    *[f"sbatch run_slurm.sh configs/acdc/night/lora-night-acdc.yaml 1 ./output/acdc/acdc-night_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME acdc-night_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
]

commands_muses = [
    ##############################################
    # *[f"sbatch run_slurm.sh configs/muses/multidomain_fft.yaml 1 ./output/muses/muses-multidomain_fullfinetune_index-{seed} MODEL.LORA.ENABLED False  MODEL.LORA.NAME muses-multidomain_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    # *[f"sbatch run_slurm.sh configs/muses/multidomain_lora.yaml 1 ./output/muses/muses-multidomain_lora_index-{seed} MODEL.LORA.NAME muses-multidomain_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    *[f"sbatch run_slurm.sh configs/muses/clear/muses-clear-day.yaml 1  ./output/muses/muses-clear-day_index-{seed} MODEL.LORA.NAME muses-clear-day_index-{seed} SEED {seed}" for seed in SEEDS],
    # *[f"sbatch run_slurm.sh configs/muses/clear/muses-clear-day.yaml 1  ./output/muses/muses-clear-day_fullfinetune_index-{seed}  MODEL.LORA.ENABLED False MODEL.LORA.NAME muses-clear-day_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    ##############################################
    *[f"sbatch run_slurm.sh configs/muses/clear/muses-clear-night.yaml 1  ./output/muses/muses-clear-night_index-{seed} MODEL.LORA.NAME muses-clear-night_index-{seed} SEED {seed}" for seed in SEEDS],
    # *[f"sbatch run_slurm.sh configs/muses/clear/muses-clear-night.yaml 1  ./output/muses/muses-clear-night_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME muses-clear-night_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    # ##############################################
    *[f"sbatch run_slurm.sh configs/muses/rain/muses-rain-day.yaml 1  ./output/muses/muses-rain-day_index-{seed} MODEL.LORA.NAME muses-rain-day_index-{seed} SEED {seed}" for seed in SEEDS],
    # *[f"sbatch run_slurm.sh configs/muses/rain/muses-rain-day.yaml 1  ./output/muses/muses-rain-day_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME muses-rain-day_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    # ##############################################
    *[f"sbatch run_slurm.sh configs/muses/rain/muses-rain-night.yaml 1  ./output/muses/muses-rain-night_index-{seed} MODEL.LORA.NAME muses-rain-night_index-{seed} SEED {seed}" for seed in SEEDS],
    # *[f"sbatch run_slurm.sh configs/muses/rain/muses-rain-night.yaml 1  ./output/muses/muses-rain-night_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME muses-rain-night_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    # ##############################################
    *[f"sbatch run_slurm.sh configs/muses/fog/muses-fog-day.yaml 1  ./output/muses/muses-fog-day_index-{seed} MODEL.LORA.NAME muses-fog-day_index-{seed} SEED {seed}" for seed in SEEDS],
    # *[f"sbatch run_slurm.sh configs/muses/fog/muses-fog-day.yaml 1  ./output/muses/muses-fog-day_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME muses-fog-day_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    # ##############################################
    *[f"sbatch run_slurm.sh configs/muses/fog/muses-fog-night.yaml 1  ./output/muses/muses-fog-night_index-{seed} MODEL.LORA.NAME muses-fog-night_index-{seed} SEED {seed}" for seed in SEEDS],
    # *[f"sbatch run_slurm.sh configs/muses/fog/muses-fog-night.yaml 1  ./output/muses/muses-fog-night_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME muses-fog-night_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    # ##############################################
    *[f"sbatch run_slurm.sh configs/muses/snow/muses-snow-day.yaml 1  ./output/muses/muses-snow-day_index-{seed} MODEL.LORA.NAME muses-snow-day_index-{seed} SEED {seed}" for seed in SEEDS],
    # *[f"sbatch run_slurm.sh configs/muses/snow/muses-snow-day.yaml 1  ./output/muses/muses-snow-day_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME muses-snow-day_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    # ##############################################
    *[f"sbatch run_slurm.sh configs/muses/snow/muses-snow-night.yaml 1  ./output/muses/muses-snow-night_index-{seed} MODEL.LORA.NAME muses-snow-night_index-{seed} SEED {seed}" for seed in SEEDS],
    # *[f"sbatch run_slurm.sh configs/muses/snow/muses-snow-night.yaml 1  ./output/muses/muses-snow-night_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME muses-snow-night_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
    # ##############################################
]

commands_bdd = [
    *[f"sbatch run_slurm.sh configs/bdd/bdd.yaml 1 ./output/bdd/bdd_index-{seed} MODEL.LORA.NAME bdd_index-{seed} SEED {seed}" for seed in SEEDS],
    *[f"sbatch run_slurm.sh configs/bdd/bdd.yaml 1 ./output/bdd/bdd_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME bdd_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
]

shard_root = os.path.join(_root, "sharded_datasets", "bdd100k",)
shards = get_sharded_names(shard_root)
commands_bdd_shards = [
    *[rf'sbatch run_slurm.sh configs/bdd/bdd.yaml 1 ./output/bdd/{name}_index-{seed} MODEL.LORA.NAME {name}_index-{seed} DATASETS.TRAIN \(\"{name}_sem_seg_train\"\,\) SEED {seed}' for seed in SEEDS[:1] for name in shards],
    # *[f"sbatch run_slurm.sh configs/bdd/bdd.yaml 1 ./output/bdd/bdd_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME bdd_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
]

commands_mv = [
    *[f"sbatch run_slurm.sh configs/mv/mv.yaml 1 ./output/mv/mv_index-{seed} MODEL.LORA.NAME mv_index-{seed} SEED {seed}" for seed in SEEDS],
    *[f"sbatch run_slurm.sh configs/mv/mv.yaml 1 ./output/mv/mv_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME mv_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
]

commands_nyu = [
    *[f"sbatch run_slurm.sh configs/nyu/nyu.yaml 1 ./output/nyu/nyu_index-{seed} MODEL.LORA.NAME nyu_index-{seed} SEED {seed}" for seed in SEEDS],
    *[f"sbatch run_slurm.sh configs/nyu/nyu.yaml 1 ./output/nyu/nyu_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME nyu_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
]

commands_a150 = [
    *[f"sbatch run_slurm_distributed.sh configs/a150/a150.yaml 2 ./output/a150/a150_index-{seed} MODEL.LORA.NAME a150_index-{seed} SEED {seed}" for seed in SEEDS],
    # *[f"sbatch run_slurm.sh configs/a150/a150.yaml 2 ./output/a150/a150_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME a150_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
]

commands_idd = [
    *[f"sbatch run_slurm.sh configs/idd/idd.yaml 1 ./output/idd/idd_index-{seed} MODEL.LORA.NAME idd_index-{seed} SEED {seed}" for seed in SEEDS],
    *[f"sbatch run_slurm.sh configs/idd/idd.yaml 1 ./output/idd/idd_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME idd_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
]

commands_pc59 = [
    *[f"sbatch run_slurm.sh configs/pc59/pc59.yaml 1 ./output/pc59/pc59_index-{seed} MODEL.LORA.NAME pc59_index-{seed} SEED {seed}" for seed in SEEDS],
    *[f"sbatch run_slurm.sh configs/pc59/pc59.yaml 1 ./output/pc59/pc59_fullfinetune_index-{seed} MODEL.LORA.ENABLED False MODEL.LORA.NAME pc59_fullfinetune_index-{seed} SEED {seed}" for seed in SEEDS],
]
command_coconutL = "sbatch run_slurm_theo.sh  configs/coconutL/coconutL.yaml 1 ./output/coconut/coconut-l_fullfinetune_index-1000 MODEL.LORA.NAME coconut-l_fullfinetune_index-1000 MODEL.LORA.ENABLED False SEED 1000"
for command in commands_a150:
    rc = subprocess.call(command, shell=True)