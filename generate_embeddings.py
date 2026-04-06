from pathlib import Path
import os
import yaml
import json
import argparse
from catseg.train_net import setup
from domain_orchestrator.utils import get_domain_args
from domain_orchestrator.embedding import OpenClipEmbeddingModel

DETECTRON2_DATASET_PATH = os.getenv("DETECTRON2_DATASETS")

def get_classnames_for_domain(domain):
    # TODO Klassennamen irgendwie anders bekommen, da auf diese Art und Weise zu viel unnötige Logs erstellt werden
    domain_args = get_domain_args(domain, "val", get_cofing_only=True)
    cfg = setup(domain_args)

    json_path = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON

    with open(json_path) as f:
        return json.load(f)

if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser()
    # Path to the yaml file that contains the paths to the domains training data
    parser.add_argument("--source_domains_file", type=str, required=True)
    # Path to the lora library where the statistics will be stored
    parser.add_argument("--lora_library_path", type=str, required=True)

    # NEU: welches CLIP soll genutzt werden?
    parser.add_argument(
        "--clip_source",
        type=str,
        required=True,
        choices=["open_clip", "huggingface"],
    )

    # Parse arguments
    args = parser.parse_args()
    source_domains_file = Path(args.source_domains_file)
    lora_library_path = Path(args.lora_library_path)
    clip_source = args.clip_source

    with open(source_domains_file, "r") as f:
        source_domains = yaml.safe_load(f)

    embedding_manager = None

    print("Generating embeddings for all source domains ...")
    
    for domain_name in source_domains:
        print(domain_name)
        args = get_domain_args(domain_name, "train", get_cofing_only=True)
        train_dataset_path = Path(args.train_dataset_path)
        #print(train_dataset_path)

        assert train_dataset_path.exists(), f"Path to training dataset {train_dataset_path} does not exist!"

        if embedding_manager is None:
            from domain_orchestrator import embedding
            if clip_source == "huggingface":
                embedding_manager = embedding.EmbeddingManager()
            elif clip_source == "open_clip":
                embedding_manager = embedding.EmbeddingManager(OpenClipEmbeddingModel())
            else:
                raise ValueError(f"Invalid clip source: {clip_source}")

        domain_path = lora_library_path / Path(domain_name)

        embedding_manager.calculate_statistics(
            domain_name=domain_name,
            domain_path=domain_path,
            train_path=train_dataset_path,
        )
        embedding_manager.calculate_vocabulary_embeddings(
            domain_name=domain_name,
            domain_path=domain_path,
            classnames=get_classnames_for_domain(domain_name)
        )

    print("Finished generating embeddings for all domains")