import os
from typing import List, Dict, Any

import yaml

from domain_orchestrator.domain_orchestrator import DomainOrchestrator, NormalizationMethod
from domain_orchestrator.embedding import VocabEmbeddingMethod
from experiments import load_domains_from_yaml, load_config_from_yaml, NAME_MEASURE_MAPPING


def qualitative_vocab_expansion(config: Dict[str, Any], target_domains,
                                top_x: int, image_paths: List[str],
                                output_dir: str, orchestrator: DomainOrchestrator,
                                vocab_embedding_method: VocabEmbeddingMethod,
                                normalization_method: NormalizationMethod) -> None:
    root_dir = os.path.abspath(os.path.dirname(__file__))
    base_output_dir = os.path.join(
        root_dir, output_dir
    )

    similarity_measure_name = config.get("similarity_measure_name", "euclidean")
    orchestrator.benchmark_semla_qualitative(
        target_domains=target_domains,
        top_k=config.get("top_k"),
        remove_target_adapter=True,
        similarity_measure=NAME_MEASURE_MAPPING[similarity_measure_name],
        softmax_temperature=config.get("temperature"),
        top_x=top_x,
        image_paths=image_paths,
        output_dir=base_output_dir,
        gamma=config.get("gamma", 0.5),
        top_q_frac=config.get("top_q_frac", 0.5),
        vocab_embedding_method=vocab_embedding_method,
        normalization_method=normalization_method,
    )

if __name__ == "__main__":
    image_list_file = "test-images.yaml"
    source_domains_path = "config/source_domains.yaml"
    target_domains_path = "config/target_domains_qualitative_examples.yaml"
    semla_config_path = "config/semla_config_best_none.yaml"
    output_dir = "./results/qualit/none/"
    image_paths = None
    top_x = 5

    target_domains = load_domains_from_yaml(target_domains_path)
    source_domains = load_domains_from_yaml(source_domains_path)
    semla_config = load_config_from_yaml(semla_config_path)
    voc_method = VocabEmbeddingMethod.NONE
    norm_method = NormalizationMethod.ZSCORE

    with open(image_list_file, "r") as f:
        image_paths = yaml.safe_load(f)

    orchestrator = DomainOrchestrator(source_domains, voc_method,
                                      normalize_centroids=True)
    qualitative_vocab_expansion(semla_config, target_domains,
                                top_x, image_paths,
                                output_dir, orchestrator,
                                voc_method, norm_method)