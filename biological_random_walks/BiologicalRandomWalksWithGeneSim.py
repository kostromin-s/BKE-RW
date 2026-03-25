import networkx as nx
import csv
from biological_random_walks.BiologicalRandomWalks import BiologicalRandomWalks


class BiologicalRandomWalksWithGeneSim(BiologicalRandomWalks):

    def __init__(self,
                 gene_similarity_file_path,
                 *args,
                 **kwargs):

        self.gene_similarity_file_path = gene_similarity_file_path

        super().__init__(*args, **kwargs)

    def load_gene_similarity_network(self):
        G = nx.Graph()

        with open(self.gene_similarity_file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header nếu có

            for row in reader:
                u, v, w = row
                G.add_edge(u, v, weight=float(w))

        return G