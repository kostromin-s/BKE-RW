import os
import csv

class KnowledgeGraph:
    def __init__(self, PPI = None, GeneTerm = None, ):
        self.PPI = PPI
        self.GeneTerm = GeneTerm
        self.edges = []
