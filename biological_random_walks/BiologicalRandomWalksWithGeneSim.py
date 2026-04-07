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

    def compute_matrix_aggregation(self, PPI_network, CO_expression_network, matrix_aggregation_policy = "convex_combination"):
        matrix_similarity_network = self.load_gene_similarity_network()
        # Cập nhật trọng số của PPI_network theo matrix_similarity_network
        for u, v, data in PPI_network.edges(data=True):
            if matrix_similarity_network.has_edge(u, v):
                data['weight'] = matrix_similarity_network[u][v]['weight']
            else:
                data['weight'] = 0.0
        
        return super().compute_matrix_aggregation(PPI_network, CO_expression_network, matrix_aggregation_policy)

    def load_gene_similarity_network(self):
        G = nx.Graph()

        with open(self.gene_similarity_file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header nếu có

            for row in reader:
                u, v, w = row
                G.add_edge(u, v, weight=float(w))

        return G