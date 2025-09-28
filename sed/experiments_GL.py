import os
from pathlib import Path
import random
import shutil
import sys
from PIL import Image
# GL Imports
import argparse
import json
import pickle


import numpy as np
import scipy
import scipy.spatial

import detectron2.utils.comm as comm

from domain_orchestrator import *

from utils import custom_domain_args, domain_args

from train_net import Trainer, setup

torch.set_float32_matmul_precision("high")

# logging.disable()

########### SETUP

NAME_MEASURE_MAPPING = {
    "euclidean": lambda u, v: scipy.spatial.distance.euclidean(
        u.squeeze(), v.squeeze()
    ),
    "cosine": lambda u, v: scipy.spatial.distance.cosine(u.squeeze(), v.squeeze()),
    "mahalanobis": lambda u, v, S: scipy.spatial.distance.mahalanobis(
        u.squeeze(), v.squeeze(), S
    ),
}

BASE_MODEL_CONFIG = custom_domain_args(
    # config_file="configs/acdc/acdc_base.yaml",
    config_file="configs/config.yaml",
    num_gpus=1,
    output_path="output/base/eval",
    model_path="models/model_final.pth",
)


def list_subdirectories(directory):
    try:
        # Create a Path object for the directory
        path = Path(directory)
        # Use the glob method to list all sub-directories
        subdirectories = [subdir.name for subdir in path.iterdir() if subdir.is_dir()]
        return subdirectories
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied for accessing {directory}.")
        return []

_DATASET_ROOT = os.getenv("DETECTRON2_DATASETS", "datasets")

#### New experiments


def benchmark_experts(config_dict=None, results_directory=None):
    config_path = f"{results_directory}/config.json"
    json.dump(config_dict, open(config_path, 'w'))

    distance_measure_name = config_dict['merge_settings']['distance_metric']
    source_domains = config_dict['source_domains']
    target_domains = config_dict['target_domains']

    orchestrator = DomainOrchestrator(
        source_domains,
        target_domains,
        distance_measure_name=distance_measure_name,
        distance_measure=NAME_MEASURE_MAPPING[distance_measure_name],
    )

    results: dict = orchestrator.benchmark_experts(is_adapter=True, zeroshot=False)

    results_path = f"{results_directory}/results.json"
    json.dump(results, open(results_path, 'w'))



def zeroshot(config_dict=None, results_directory=None):
    config_path = f"{results_directory}/config.json"
    json.dump(config_dict, open(config_path, 'w'))
    distance_measure_name = config_dict['merge_settings']['distance_metric']
    source_domains = config_dict['source_domains']
    target_domains = config_dict['target_domains']

    orchestrator = DomainOrchestrator(
        source_domains,
        target_domains,
        distance_measure_name=distance_measure_name,
        distance_measure=NAME_MEASURE_MAPPING[distance_measure_name],
    )

    results: dict = orchestrator.benchmark_experts(is_adapter=False, zeroshot=True)

    results_path = f"{results_directory}/results.json"
    json.dump(results, open(results_path, 'w'))
 


def online_merge_full_GL(config_dict=None, results_directory=None):
    config_path = f"{results_directory}/config.json"
    json.dump(config_dict, open(config_path, 'w'))
    
    # Merge parameters
    distance_measure_name = config_dict['merge_settings']['distance_metric']
    temperature = config_dict['merge_settings']['temperature']
    window_size = config_dict['merge_settings']['window_size']
    distance_thresh = config_dict['merge_settings']['distance_thresh']
    k_adapters = config_dict['merge_settings']['top_k']

    # Remove target adapter and full model merging
    remove_target_adapter = bool(config_dict['remove_target'])
    full_model_merging = True
    
    # Source and target domains
    source_domains = config_dict['source_domains']
    target_domains = config_dict['target_domains']

    orchestrator = DomainOrchestrator(
        source_domains,
        target_domains,
        distance_measure_name=distance_measure_name,
        distance_measure=NAME_MEASURE_MAPPING[distance_measure_name],
        temp=temperature,
        k_adapters=k_adapters,
        no_lora=True
    )

    results, weights = orchestrator.online_centroid_merge_full(
        remove_target_adapter=remove_target_adapter,
        full_model_merging=True  
    )

    # Save data to pickle files - results and weights
    results_path = f"{results_directory}/results.json"
    json.dump(results, open(results_path, 'w'))
    weights_path = f"{results_directory}/weights.json"
    json.dump(weights, open(weights_path, 'w'))
 



def uniform_merge_GL(config_dict=None, results_directory=None):
    config_path = f"{results_directory}/config.json"
    json.dump(config_dict, open(config_path, 'w'))

    # Merge parameters
    distance_measure_name = config_dict['merge_settings']['distance_metric']
    temperature = config_dict['merge_settings']['temperature']
    window_size = config_dict['merge_settings']['window_size']
    distance_thresh = config_dict['merge_settings']['distance_thresh']
    k_adapters = len(config_dict['source_domains'])

    # Remove target adapter and full model merging
    remove_target_adapter = bool(config_dict['remove_target'])
    full_model_merging = False

    # Source and target domains
    source_domains = config_dict['source_domains']
    target_domains = config_dict['target_domains']


    orchestrator = DomainOrchestrator(
        source_domains,
        target_domains,
        distance_measure_name=distance_measure_name,
        distance_measure=NAME_MEASURE_MAPPING[distance_measure_name],
        temp=temperature,
        window_size=window_size,
        distance_thresh=distance_thresh,
        k_adapters=k_adapters,
    )

    results, weights = orchestrator.batch_uniform_merge(
        remove_target_adapter=remove_target_adapter,
        full_model_merging=full_model_merging
    )
    print(results)
    print(weights)

    # Save data to pickle files - results and weights
    results_path = f"{results_directory}/results.json"
    json.dump(results, open(results_path, 'w'))
    weights_path = f"{results_directory}/weights.json"
    json.dump(weights, open(weights_path, 'w'))



def online_merge_GL(config_dict=None, results_directory=None):
    config_path = f"{results_directory}/config.json"
    json.dump(config_dict, open(config_path, 'w'))

    # Merge parameters
    distance_measure_name = config_dict['merge_settings']['distance_metric']
    temperature = config_dict['merge_settings']['temperature']
    window_size = config_dict['merge_settings']['window_size']
    distance_thresh = config_dict['merge_settings']['distance_thresh']
    k_adapters = config_dict['merge_settings']['top_k']

    # Remove target adapter and full model merging
    remove_target_adapter = bool(config_dict['remove_target'])
    full_model_merging = False

    # Source and target domains
    source_domains = config_dict['source_domains']
    target_domains = config_dict['target_domains']

    orchestrator = DomainOrchestrator(
        source_domains,
        target_domains,
        distance_measure_name=distance_measure_name,
        distance_measure=NAME_MEASURE_MAPPING[distance_measure_name],
        temp=temperature,
        k_adapters=k_adapters,
    )

    results, weights = orchestrator.online_centroid_merge(
        remove_target_adapter=remove_target_adapter,
        full_model_merging=full_model_merging
    )


    # Save data to pickle files - results and weights
    results_path = f"{results_directory}/results.json"
    json.dump(results, open(results_path, 'w'))
    weights_path = f"{results_directory}/weights.json"
    json.dump(weights, open(weights_path, 'w'))
    config_path = f"{results_directory}/config.json"
    json.dump(config_dict, open(config_path, 'w'))
    


# new_uniform_merge(remove_target_adapter=False)

# centroid_merge(remove_target_adapter=False, full_model_merging=False)

# online_merge(remove_target_adapter=False, full_model_merging=False)


# TEST 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config_file", type=str, help="Path to the JSON config file.")
    parser.add_argument("results_directory", type=str, help="Path to the results folder.")
    args = parser.parse_args()
    config_dict = json.load(open(args.config_file, 'r'))
    results_directory = args.results_directory

    if config_dict['merge_settings']['merge_strategy'] == 'online_merge':
        print(config_dict)
        online_merge_GL(config_dict=config_dict, results_directory=results_directory)
    elif config_dict['merge_settings']['merge_strategy'] == 'uniform_merge':
        print(config_dict)
        uniform_merge_GL(config_dict=config_dict, results_directory=results_directory)
    elif config_dict['merge_settings']['merge_strategy'] == 'benchmark_experts':
        print(config_dict)
        benchmark_experts(config_dict=config_dict, results_directory=results_directory)
    elif config_dict['merge_settings']['merge_strategy'] == 'zeroshot':
        print(config_dict)
        zeroshot(config_dict=config_dict, results_directory=results_directory)
    elif config_dict['merge_settings']['merge_strategy'] == 'online_merge_full':
        print(config_dict)
        online_merge_full_GL(config_dict=config_dict, results_directory=results_directory)

    print("Done")
       

"""

# Load configuration from JSON
log_directory_parent=$(jq -r '.results_folder_parent' "$1")
log_directory_results="$log_directory_parent/$(jq -r '.experiment_name' "$1")"

# Create directories if they don't already exist
mkdir -p "$log_directory_parent"
mkdir -p "$log_directory_results"

# Define output and error file paths and export as environment variables
export SLURM_OUTPUT_FILE="$log_directory_results/output.txt"
export SLURM_ERROR_FILE="$log_directory_results/error.txt"

"""