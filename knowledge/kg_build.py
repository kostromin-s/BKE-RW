import os
import hydra
from omegaconf import DictConfig
import csv


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


ROOT_DIR = get_project_root()


def load_gene_term(file_path):
    edges = []

    with open(file_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            gene = row["gene_id"]
            term = row["term_id"]
            db = row["DB"]

            if db == "Reactome":
                relation = "involved_in"
                term_type = "Pathway"

            elif db == "GO":
                relation = "annotated_with"
                term_type = "GO"

            else:
                continue

            edges.append({
                "head": gene,
                "relation": relation,
                "tail": term,
                "head_type": "Gene",
                "tail_type": term_type
            })

    return edges


def load_ppi(file_path):
    edges = []

    with open(file_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            u = row["u"]
            v = row["v"]

            # undirected → thêm 2 chiều
            edges.append({
                "head": u,
                "relation": "interacts_with",
                "tail": v,
                "head_type": "Gene",
                "tail_type": "Gene"
            })

            edges.append({
                "head": v,
                "relation": "interacts_with",
                "tail": u,
                "head_type": "Gene",
                "tail_type": "Gene"
            })

    return edges


def save_kg(edges, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow([
            "head", "relation", "tail",
            "head_type", "tail_type"
        ])

        for e in edges:
            writer.writerow([
                e["head"],
                e["relation"],
                e["tail"],
                e["head_type"],
                e["tail_type"]
            ])


@hydra.main(
    version_base=None,
    config_path=os.path.join(ROOT_DIR, "config"),
    config_name="kg_build"
)
def build_kg(cfg: DictConfig):

    print("Building the knowledge graph with config:")
    print(cfg)

    gene_term_path = cfg.data.gene_term
    ppi_path = cfg.data.ppi
    output_path = cfg.output.path

    print("\nLoading Gene–Term...")
    gene_term_edges = load_gene_term(gene_term_path)

    print("Loading PPI...")
    ppi_edges = load_ppi(ppi_path)

    print("\nMerging edges...")
    all_edges = gene_term_edges + ppi_edges

    print(f"Total edges: {len(all_edges)}")

    print("\nSaving KG...")
    save_kg(all_edges, output_path)

    print(f"KG saved to: {output_path}")


if __name__ == "__main__":
    build_kg()