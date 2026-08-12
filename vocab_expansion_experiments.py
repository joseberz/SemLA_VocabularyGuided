from typing import List, Dict, Any

import yaml

from domain_orchestrator.domain_orchestrator import DomainOrchestrator, NormalizationMethod
from domain_orchestrator.embedding import VocabEmbeddingMethod
from experiments import load_domains_from_yaml, load_config_from_yaml


def qualitative_vocab_expansion(config: Dict[str, Any], target_domains,
                                top_x: int, image_paths: List[str],
                                output_dir: str, orchestrator: DomainOrchestrator,
                                vocab_embedding_method: VocabEmbeddingMethod,
                                normalization_method: NormalizationMethod) -> None:
    orchestrator.benchmark_semla_qualitative(
        target_domains=target_domains,
        top_x=top_x,
        image_paths=image_paths,
        output_dir=output_dir,
        top_k=config.get("top_k", 5),
        gamma=config.get("gamma", 0.5),
        top_q_frac=config.get("top_q_frac", 0.5),
        vocab_embedding_method=vocab_embedding_method,
        normalization_method=normalization_method,
    )

if __name__ == "__main__":
    image_list_file = ""
    source_domains_path = ""
    target_domains_path = ""
    semla_config_path = ""
    output_dir = ""
    image_paths = None
    top_x = 3

    target_domains = load_domains_from_yaml(target_domains_path)
    source_domains = load_domains_from_yaml(source_domains_path)
    semla_config = load_config_from_yaml(semla_config_path)
    voc_method = VocabEmbeddingMethod.PATCH
    norm_method = NormalizationMethod.ZSCORE

    with open(image_list_file, "r") as f:
        image_paths = yaml.safe_load(f)

    orchestrator = DomainOrchestrator(source_domains, voc_method,
                                      normalize_centroids=True)
    qualitative_vocab_expansion(semla_config, target_domains,
                                top_x, image_paths,
                                output_dir, orchestrator,
                                voc_method, norm_method)