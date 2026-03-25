from hydra.core.hydra_config import HydraConfig
import os
import hydra
from omegaconf import DictConfig
from biological_random_walks.BiologicalRandomWalks import BiologicalRandomWalks
from biological_random_walks.BiologicalRandomWalksWithGeneSim import BiologicalRandomWalksWithGeneSim

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    run_dir = HydraConfig.get().runtime.output_dir
    output_path = os.path.join(run_dir, "result.txt")

    if cfg.method == "gene_sim":

        brw = BiologicalRandomWalksWithGeneSim(

            gene_similarity_file_path = cfg.paths.gene_similarity,

            seed_file_path=cfg.experiment.seed,
            secondary_seed_file_path=cfg.experiment.de,

            ppi_file_path=cfg.paths.ppi,
            co_expression_file_path=cfg.experiment.coexpr,

            disease_ontology_file_path=cfg.experiment.disease_ontology,
            map__gene__ontologies_file_path=cfg.paths.ontology_network,

            restart_prob=cfg.params.restart_prob,
            alpha=cfg.params.alpha,
            beta=cfg.params.beta,

            network_weight_flag=False,

            output_file_path=output_path
        )

    else:  # original

        brw = BiologicalRandomWalks(

            seed_file_path=cfg.experiment.seed,
            secondary_seed_file_path=cfg.experiment.de,

            ppi_file_path=cfg.paths.ppi,
            co_expression_file_path=cfg.experiment.coexpr,

            disease_ontology_file_path=cfg.experiment.disease_ontology,
            map__gene__ontologies_file_path=cfg.paths.ontology_network,

            restart_prob=cfg.params.restart_prob,
            alpha=cfg.params.alpha,
            beta=cfg.params.beta,

            output_file_path=output_path
        )

if __name__ == "__main__":
    main()
    