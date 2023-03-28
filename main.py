import numpy as np
import networkx as nx
from pathlib import Path
from geopy.distance import great_circle
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from pytrails.hyptrails import MarkovChain

from mycode.Metrics import HYDRASMETRICS
from mycode.SyntheticDatasets import *
from mycode.BibliographicDataset import *
from mycode.Visualization import plot_hyptrails


class Hydras:
    def __init__(self, args: dict) -> None:
        self.confidence_factors = args['ks']
        self.network = args['network']
        self.metrics = args['metrics']
        self.transitions = nx.adjacency_matrix(self.network)
        self.hypotheses = self.create_hypotheses()
        self.evidence = self.run_hyptrails(self.transitions, self.hypotheses)
        for k, v in self.evidence.items():
            print(f"{k}: {v}")
        plot_hyptrails(self.evidence, k_values=self.confidence_factors, save_path=Path("data", "images", "test.png"))

    def create_hypotheses(self) -> dict:
        hypotheses = {x: csr_matrix(self.transitions.shape) for x in self.metrics.keys()}
        for name, metric in self.metrics.items():
            attributes = {node: self.network.nodes[node][name] for node in self.network.nodes()}
            for row in tqdm(self.network.nodes(), desc=f"Creating hypothesis for {name}"):
                sim_row = [self.calculate_similarity(attributes[row], attributes[col], metric) for col in self.network.nodes()]
                if sum(sim_row) == 0:
                    hypotheses[name][row] = csr_matrix(sim_row)
                else:
                    hypotheses[name][row] = csr_matrix(sim_row) / sum(sim_row)
        return hypotheses

    @staticmethod
    def calculate_similarity(x, y, metric: HYDRASMETRICS) -> float:
        if metric == HYDRASMETRICS.EQUALS:
            return 1 if x == y else 0
        elif metric == HYDRASMETRICS.OVERLAP:
            return len(set(x) & set(y))
        elif metric == HYDRASMETRICS.GEO_DISTANCE:
            return great_circle(x, y).km
        elif metric == HYDRASMETRICS.COSINE_SIMILARITY:
            return distance.cosine(x, y)
        elif metric == HYDRASMETRICS.DISTANCE:
            return abs(x - y)
        else:
            print(f"Error: No metric implemented for {str(metric)}. ")
            exit(1)

    def run_hyptrails(self, transitions: csr_matrix, hypotheses: dict) -> dict:
        evidence = {}
        for name, hypothesis in hypotheses.items():
            evidence[name] = [MarkovChain.marginal_likelihood(transitions, hypothesis * ks) for ks in self.confidence_factors]
        return evidence
                

if __name__ == '__main__':
    SYNTHETIC = False
    if SYNTHETIC:
        Hydras(args={
            'network': SyntheticDataset(args={
                'size': 100,
                'connections': 5,
                'attributes': {
                    'age': [(np.random.choice, np.arange(18,65))],  # rand int between
                    'gender': [(np.random.choice, ['m', 'f'])],  # either one or another
                    'affiliations': [(np.random.choice, UNIVERSITIES, np.random.randint(1, 5))],  # one to 5 affiliations todo fix that it generates individually
                    'geo_coords': [(np.random.uniform, *COORDS['lat']), (np.random.uniform, *COORDS['long'])],  # generate random geo coords within germany
                    'semantic_representation': [(np.random.rand, 768)]  # create a random vector representation
                } 
            }).get_network(),
            'metrics': {'age': HYDRASMETRICS.EQUALS, 'gender': HYDRASMETRICS.EQUALS, 'affiliations': HYDRASMETRICS.OVERLAP, 'geo_coords': HYDRASMETRICS.GEO_DISTANCE, 'semantic_representation': HYDRASMETRICS.COSINE_SIMILARITY},
            'ks': [1, 3, 5, 10, 50, 100, 1000, 10000]
        })
    else:
        Hydras(args={
            'network': BibliographicDataset(args={
                "path": Path("data", "bibliometric_dataset"),
                "separation_year": 2016, 
                'verbose': 1
            }).get_network(), 
            'metrics': {
                'prev_co_authors': HYDRASMETRICS.OVERLAP,  # social
                'graph_vectors': HYDRASMETRICS.COSINE_SIMILARITY,  # social
                'semantic_vectors': HYDRASMETRICS.COSINE_SIMILARITY,  # social
                'citations':HYDRASMETRICS.OVERLAP,  # cognitive
                'venue':HYDRASMETRICS.OVERLAP,  # cognitive
                's2fieldsofstudy': HYDRASMETRICS.OVERLAP,  #cognitive
                'schol_affiliations': HYDRASMETRICS.OVERLAP,  # organisational
                'dblp_affiliations': HYDRASMETRICS.OVERLAP,  # organisational
                'countries': HYDRASMETRICS.OVERLAP,  # organisational
                'schol_hindex': HYDRASMETRICS.DISTANCE,  # institutional
                'country_geocodes': HYDRASMETRICS.GEO_DISTANCE,  # geographic 
                'affi_geocodes': HYDRASMETRICS.GEO_DISTANCE,  # geographic
            }, 
            'ks': [1, 3, 5, 10, 50, 100, 1000, 10000]
        })
