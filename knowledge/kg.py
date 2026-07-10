import os
import csv

class KnowledgeGraph:
    def __init__(self, PPI = None, GeneTerm = None, nameDataset = None):
        self.PPI = PPI
        self.GeneTerm = GeneTerm
        self.nameDataset = nameDataset
        self.edges = []

    def edge_gene_term(self):
        if self.GeneTerm is None:
            print("No Gene–Term data provided.")
            return
        for row in self.GeneTerm.itertuples(index=False):
            geneId = row.gene_id
            termID = row.term_id
            DB = row.DB
            relation = None
            tail_type = None

            if DB == "GO":
                relation = "annotated_with"
                tail_type = "GO"

            elif DB == "Reactome":
                relation = "involved_in"
                tail_type = "Pathway"
            
            else:
                continue

            self.edges.append({
                "head": geneId,
                "relation": relation,
                "tail": termID,
                "head_type": "Gene",
                "tail_type": tail_type
            })

    def edge_ppi(self):
        if self.PPI is None:
            print("No PPI data provided.")
            return
        for row in self.PPI.itertuples(index=False):
            u = row.u
            v = row.v
            # undirected → thêm 2 chiều
            self.edges.append({
                "head": u,
                "relation": "interacts_with",
                "tail": v,
                "head_type": "Gene",
                "tail_type": "Gene"
            })

            self.edges.append({
                "head": v,
                "relation": "interacts_with",
                "tail": u,
                "head_type": "Gene",
                "tail_type": "Gene"
            })
    
    def save_edges(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(
            base_dir,
            f"KGs/{self.nameDataset}_kg.tsv"
        )
        with open(output_path, "w", newline='') as f:
            fieldnames = ["head", "relation", "tail", "head_type", "tail_type"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for edge in self.edges:
                writer.writerow(edge)

    def build_kg(self):
        self.edge_gene_term()
        self.edge_ppi()
        self.save_edges()