import os
import hydra
import torch
import csv
import pandas as pd

from knowledge.kg import KnowledgeGraph
from knowledge.rgcn import Preprocessor


def load_file(file_path):
    df = pd.read_csv(file_path, delimiter="\t")
    return df

def run_mode_1(PPI, GeneTerm, DO, nameDataset):
    print("Building the knowledge graph...")
    employee = KnowledgeGraph(PPI, GeneTerm, nameDataset)
    employee.build_kg()
    print("Knowledge graph built and saved successfully.")

def run_mode_2(KG, hidden_dim=64, dropout=0.3, nameDataset="", k=10, epochs=20, lr=0.01, num_neg=5):
    print("Processing the knowledge graph...")

    employee = Preprocessor(KG,
                            hidden_dim,
                            dropout,
                            nameDataset,
                            k=k,
                            epochs=epochs,
                            lr=lr,
                            num_neg=num_neg)
    employee.run()
    print("Knowledge graph processed successfully.")

@hydra.main(version_base=None, config_path="config", config_name="kg_build")
def main(cfg):
    print("Loading data...")
    ppi_path = cfg.data.ppi
    gene_term_path = cfg.data.gene_term
    do_path = cfg.experiment.disease_ontology
    nameDataset = cfg.experiment.name.upper()
    kg_path = cfg.experiment.kg
    PPI = []
    GeneTerm = []
    DO = []
    KG = []
    mode = cfg.mode

    if mode == 1 or mode == 3:
        if not os.path.exists(ppi_path) or not os.path.exists(gene_term_path) or not os.path.exists(do_path):
            print(f"Error: Files {ppi_path if not os.path.exists(ppi_path) else gene_term_path if not os.path.exists(gene_term_path) else do_path} do not exist. Please provide the necessary files before running the program.")
            return
        print("Loading PPI and Gene-Term data...")
        PPI = load_file(ppi_path)
        print(f"Loaded {len(PPI)} PPI edges.")
        GeneTerm = load_file(gene_term_path)
        print(f"Loaded {len(GeneTerm)} Gene-Term edges.")
        DO = load_file(do_path)
        print(f"Loaded {len(DO)} DO edges.")

        valid_terms = set(DO["Term_ID"])
        GeneTerm = GeneTerm[GeneTerm["term_id"].isin(valid_terms)]

    if mode == 2:
        if not os.path.exists(kg_path):
            print(f"Error: File {kg_path} does not exist. Please provide the KG file before running the program.")
            return
        print("Loading KG data...")
        KG = load_file(kg_path)
        print(f"Loaded {len(KG)} KG edges.")
    
    if mode == 1:
        run_mode_1(PPI, GeneTerm, DO, nameDataset)
    elif mode == 2:
        run_mode_2(KG, 
                   hidden_dim=64, 
                   dropout=0.3, 
                   nameDataset=nameDataset, 
                   k=10, 
                   epochs=20, 
                   lr=0.01, 
                   num_neg=5)
    elif mode == 3:
        run_mode_1(PPI, GeneTerm, DO, nameDataset)

        if not os.path.exists(kg_path):
            print(f"Error: File {kg_path} does not exist. Please provide the KG file before running the program.")
            return
        print("Loading KG data...")
        KG = load_file(kg_path)
        print(f"Loaded {len(KG)} KG edges.")

        run_mode_2(KG, 
                   hidden_dim=64, 
                   dropout=0.3, 
                   nameDataset=nameDataset, 
                   k=10, 
                   epochs=20, 
                   lr=0.01, 
                   num_neg=5)

if __name__ == "__main__":
    main()