from pathlib import Path
import yaml
import argparse

from domain_orchestrator.model_adapters import CatSegAdapter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_domains_file", type=str, required=True)
    parser.add_argument("--lora_library_path", type=str, required=True)

    args = parser.parse_args()
    source_domains_file = Path(args.source_domains_file)
    lora_library_path = Path(args.lora_library_path)

    with open(source_domains_file, "r") as f:
        source_domains = yaml.safe_load(f)

    embedding_manager = None

    adapter = CatSegAdapter()
    adapter.setup()

    print("Generating embeddings for all source domains ...")

    for domain_name in source_domains:
        resources = adapter.get_domain_resources(domain_name, "train", load_data=False)
        train_dataset_path = resources.train_dataset_path

        if train_dataset_path is None or not train_dataset_path.exists():
            raise FileNotFoundError(
                f"Path to training dataset {train_dataset_path} does not exist!"
            )

        print(train_dataset_path)

        if embedding_manager is None:
            from domain_orchestrator import embedding

            embedding_manager = embedding.EmbeddingManager()

        domain_path = lora_library_path / Path(domain_name)

        embedding_manager.calculate_statistics(
            domain_name=domain_name,
            domain_path=domain_path,
            train_path=train_dataset_path,
        )

    print("Finished generating embeddings for all domains")
