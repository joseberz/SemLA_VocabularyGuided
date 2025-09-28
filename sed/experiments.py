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
import matplotlib.pyplot as plt
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

def benchmark_experts():

    distance_measure_name = "euclidean"
    source_domains = ["nyu"]
    target_domains = ["nyu"]

    orchestrator = DomainOrchestrator(
        source_domains,
        target_domains,
        distance_measure_name=distance_measure_name,
        distance_measure=NAME_MEASURE_MAPPING[distance_measure_name],
    )

    res: dict = orchestrator.benchmark_experts(is_adapter=False, zeroshot=True)

    print(res)

    # run = wandb.init(
    #     mode="offline",
    #     project="acdc_adapter_performance_across_domains",
    #     name=run_name,
    #     config={
    #         "distance_measure_name": distance_measure_name,
    #         "temperature": temperature,
    #         "window_size": window_size,
    #         "distance_thresh": distance_thresh,
    #         "k_adapters": k_adapters,
    #         "index": index,
    #     },
    # )

    # for dataset, result in res.items():
    #     print(f"Tested on {dataset}")
    #     print(f"Result: {result}")

    #     wandb.log({"mIoU": result})

    # wandb.finish()


def uniform_merge(remove_target_adapter: bool, full_model_merging: bool):

    distance_measure_name = "euclidean"
    temperature = 0.0
    window_size = None
    distance_thresh = None
    

    # source_domains = ["acdc-rain", "acdc-fog", 'acdc-snow', 'acdc-night']
    source_domains =  ['cs-normal','idd']
    target_domains =  ['acdc-rain']

    k_adapters = len(source_domains)

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

    # for result in results:
    #     dataset, index, mIoU = result

    #     run = wandb.init(
    #         mode="offline",
    #         project="cs-acdc-on-muses_batch_uniform_merge",
    #         name=f"{dataset}_{index}{('_' + 'masked') if not include_target_adapter else ''}",
    #         config={
    #             "distance_measure_name": distance_measure_name,
    #             "temperature": temperature,
    #             "window_size": window_size,
    #             "distance_thresh": distance_thresh,
    #             "k_adapters": k_adapters,
    #             "index": index,
    #             "adapters": list(source_domains)
    #         },
    #     )

    #     for domain, result in results.items():
    #         print(f"Tested on {domain}")
    #         print(f"Result: {result}")
    #         wandb.log({"mIoU": result})

    #     wandb.finish()


def centroid_merge(remove_target_adapter: bool, full_model_merging: bool, config_dict=None):

    distance_measure_name = "euclidean"
    temperature = 0.5

    source_domains = [
        "cs-normal",
        "acdc-rain",
        # "acdc-fog",
        # "acdc-snow",
        # "acdc-night",
        "muses-clear-day",
        # "muses-clear-night",
        # "muses-rain-day",
        # "muses-rain-night",
        # "muses-fog-day",
        # "muses-fog-night",
        # "muses-snow-day",
        # "muses-snow-night",
        'bdd',
        # 'mv'
    ]

    target_domains = [
        # "cs-normal",
        "acdc-rain",
        # "acdc-fog",
        # "acdc-snow",
        # "acdc-night",
        # "muses-clear-day",
        # "muses-clear-night",
        # "muses-rain-day",
        # "muses-rain-night",
        # "muses-fog-day",
        # "muses-fog-night",
        # "muses-snow-day",
        # "muses-snow-night",
        'bdd'
    ]

    # source_domains = ["acdc-rain", "acdc-fog", 'acdc-snow', 'acdc-night']
    # target_domains = ["acdc-rain", "acdc-fog", 'acdc-snow', 'acdc-night']
    
    k_adapters = len(source_domains)

    orchestrator = DomainOrchestrator(
        source_domains,
        target_domains,
        distance_measure_name=distance_measure_name,
        distance_measure=NAME_MEASURE_MAPPING[distance_measure_name],
        temp=temperature,
        k_adapters=k_adapters,
    )

    results, weights = orchestrator.batch_centroid_merge(
        remove_target_adapter=remove_target_adapter,
        full_model_merging=full_model_merging  
    )

    print(results)
    print(weights)


def online_merge(remove_target_adapter: bool, full_model_merging: bool,):

    distance_measure_name = "euclidean"
    temperature = 0.05

    source_domains =  ['cs-normal','acdc-rain','muses-snow-day','mv','bdd','a150']
    target_domains =  ['cs-normal','acdc-rain','muses-snow-day','mv','bdd','a150']
    
    k_adapters = 6

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

    print(results)
    print(weights)
    
def online_merge_test(remove_target_adapter: bool, full_model_merging: bool,):

    distance_measure_name = "euclidean"
    temperature = 0.05

    source_domains =  ['nyu']
    target_domains =  ['acdc-rain']

    k_adapters = 1

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

    print(results)
    print(weights)

def online_merge_full(remove_target_adapter: bool, full_model_merging: bool,):

    distance_measure_name = "euclidean"
    temperature = 0.05

    source_domains =  ['cs-normal','muses-clear-day',]
    target_domains =  ['acdc-rain']

    k_adapters = len(source_domains)

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

    print(results)
    print(weights)
    

def benchmark_experts():

    distance_measure_name = "euclidean"
    source_domains = ["coconutL"]
    target_domains = ["coconutL"]

    orchestrator = DomainOrchestrator(
        source_domains,
        target_domains,
        distance_measure_name=distance_measure_name,
        distance_measure=NAME_MEASURE_MAPPING[distance_measure_name],
    )

    res: dict = orchestrator.benchmark_experts(is_adapter=True, zeroshot=False)

    print(res)
    


# online_merge_full(remove_target_adapter=False, full_model_merging=True)
# uniform_merge(remove_target_adapter=False, full_model_merging=False)
benchmark_experts()
