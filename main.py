import time
import joblib
import numpy as np
import networkx as nx
from pathlib import Path
from functools import partial
from geopy.distance import great_circle
from scipy.sparse import csr_matrix, vstack
from scipy.spatial import distance
from pytrails.hyptrails import MarkovChain

from mycode.Metrics import HYDRASMETRICS
from mycode.SyntheticDatasets import *
from mycode.BibliographicDataset import *
from mycode.Visualization import plot_hyptrails

def calculate_row_similarity(row, attributes: dict, metric: HYDRASMETRICS, top_k: int) -> csr_matrix:
    current = attributes[row]
    if current is None or (isinstance(current, list) and len(current) == 0):
        return row, csr_matrix((1, len(attributes.keys())))
    if top_k:
        if metric == HYDRASMETRICS.EQUALS:
            curr = [(idy, 1 if current == val and val is not None else 0) for idy, val in attributes.items()]
        elif metric == HYDRASMETRICS.OVERLAP:
            curr = [(idy, len(current & val) if val is not None else 0) for idy, val in attributes.items()]
        elif metric == HYDRASMETRICS.GEO_DISTANCE:
            curr = [(idy, min(great_circle(x, y).km for x in current for y in val) if val is not None and len(val) != 0 else 0) for idy, val in attributes.items()]
        elif metric == HYDRASMETRICS.COSINE_SIMILARITY:
            curr = [(idy, 1 - distance.cosine(current, val)) for idy, val in attributes.items()]
        elif metric == HYDRASMETRICS.DISTANCE:
            curr = [(idy, abs(current - val) if val is not None else 0) for idy, val in attributes.items()]
        else:
            print(f"Error: No metric implemented for {str(metric)}. ")
            exit(1)
        curr = sorted(curr, key=lambda x: x[1], reverse=True)[:top_k]
        return_csr = csr_matrix((1, len(attributes.keys())))
        for col, sim in curr:
            return_csr[0, col] = sim
        return row, return_csr
    if metric == HYDRASMETRICS.EQUALS:
        curr = [1 if current == val and val is not None else 0 for val in attributes.values()]
    elif metric == HYDRASMETRICS.OVERLAP:
        curr = [len(current & val) if val is not None else 0 for val in attributes.values()]
    elif metric == HYDRASMETRICS.GEO_DISTANCE:
        curr = [min(great_circle(x, y).km for x in current for y in val) if val is not None and len(val) != 0 else 0 for val in attributes.values()]    
    elif metric == HYDRASMETRICS.COSINE_SIMILARITY:
        curr = [1 - distance.cosine(current, val) if val is not None else 0 for val in attributes.values()]
    elif metric == HYDRASMETRICS.DISTANCE:
        curr = [abs(current - val) if val is not None else 0 for val in attributes.values()]
    else:
        print(f"Error: No metric implemented for {str(metric)}. ")
        exit(1)
    return row, csr_matrix(curr)


class Hydras:
    def __init__(self, args: dict) -> None:
        self.confidence_factors = args['ks']
        self.run_name = args['run_name']
        self.network = args['network']
        self.metrics = args['metrics']
        self.calculate_matrices = args['calculate_matrices']
        self.save_matrices_path = args['save_matrices_path']
        self.calculate_evidences = args['calculate_evidences']
        self.save_evidences_path = args['save_evidences_path']
        self.build_plot = args['build_plot']
        self.verbose = args['verbose']
        if self.calculate_matrices:
            self.transitions = nx.adjacency_matrix(self.network)
            self.top_k = int(self.transitions.shape[0] * args['top_k_percentage'])
            self.nodes_to_id = {node: i for i, node in enumerate(self.network.nodes())}
            self.hypotheses = self.create_hypotheses()
            if self.verbose:
                print(f"Saving hypotheses. ")
            for name, matrix in self.hypotheses.items():
                np.save(Path(self.save_matrices_path, f"{self.run_name}-{name}.npy"), matrix)
            np.save(Path(self.save_matrices_path, f"{self.run_name}-transitions.npy"), self.transitions)
        else:
            ...
        if self.calculate_matrices:
            self.evidence = self.run_hyptrails(self.transitions, self.hypotheses)
            if self.verbose:
                print(f"Saving evidences. ")
                with open(Path(self.save_evidences_path, f"{self.run_name}.json"), "w") as f:
                    json.dump(self.evidence, f)
        else:
            self.evidence = json.load(open(Path(self.save_evidences_path, f"{self.run_name}.json"), 'r'))
        if self.verbose:
            for k, v in self.evidence.items():
                print(f"{k}: {v}")
        if self.build_plot:
            plot_hyptrails(self.evidence, k_values=self.confidence_factors, save_path=Path("data", "images", f"{self.run_name}.png"))

    @staticmethod
    def save_normalize(matrix: csr_matrix) -> csr_matrix:
        for row in matrix:
            if row.sum() != 0:
                row /= row.sum()
        return matrix

    def create_hypotheses(self) -> dict:
        hypotheses = {x: csr_matrix(self.transitions.shape) for x in self.metrics.keys()}
        for name, metric in self.metrics.items():
            print(f"\tCalculating {name} ...")
            attributes = {self.nodes_to_id[node]: self.network.nodes[node][name] for node in self.network.nodes()}
            partial_similarity = partial(calculate_row_similarity, metric=metric, attributes=attributes, top_k=self.top_k)
            tmp_hypotheses = {}
            start = time.time()
            with joblib.Parallel(n_jobs=WORKER, verbose=10) as parallel:  
                for row, sim_row in parallel(joblib.delayed(partial_similarity)(i) for i in range(len(self.network.nodes()))):
                    tmp_hypotheses[row] = sim_row
            whole = vstack(tmp_hypotheses[i] for i in range(len(self.network.nodes())))
            hypotheses[name] = whole
            if metric == HYDRASMETRICS.GEO_DISTANCE or metric == HYDRASMETRICS.DISTANCE:
                curr_max = hypotheses[name].max()
                new_scores = 1 - hypotheses[name].data / curr_max
                hypotheses[name].data = new_scores
            hypotheses[name] = self.save_normalize(hypotheses[name])
            print(f"\tDone in {time.time() - start} seconds. ")
        return hypotheses

    def run_hyptrails(self, transitions: csr_matrix, hypotheses: dict) -> dict:
        evidence = {}
        for name, hypothesis in hypotheses.items():
            evidence[name] = [MarkovChain.marginal_likelihood(transitions, hypothesis * ks) for ks in self.confidence_factors]
        return evidence
                
WORKER = 30

if __name__ == '__main__':
    SYNTHETIC = False
    if SYNTHETIC:
        Hydras(args={
            'network': SyntheticDataset(args={
                'size': 1000,
                'connections': 5,
                'attributes': {
                    'age': [(np.random.choice, np.arange(18,65))],  # rand int between
                    'gender': [(np.random.choice, ['m', 'f'])],  # either one or another
                    'affiliations': [(np.random.choice, UNIVERSITIES, np.random.randint(1, 5))],  # one to 5 affiliations todo fix that it generates individually
                    'geo_coords': [(np.random.uniform, *COORDS['lat']), (np.random.uniform, *COORDS['long'])],  # generate random geo coords within germany
                    'semantic_representation': [(np.random.rand, 768)]  # create a random vector representation
                } 
            }).get_network(),
            'top_k_percentage': 0.1,
            'metrics': {'age': HYDRASMETRICS.EQUALS, 'gender': HYDRASMETRICS.EQUALS, 'affiliations': HYDRASMETRICS.OVERLAP, 'geo_coords': HYDRASMETRICS.GEO_DISTANCE, 'semantic_representation': HYDRASMETRICS.COSINE_SIMILARITY},
            'ks': [1, 3, 5, 10, 50, 100, 1000, 10000]
        })
    else:
        Hydras(args={
            'run_name': 'ai-domain',
            'network': BibliographicDataset(args={
                'path': Path("data", "bibliometric_dataset"),
                'conferences': set(["NIPS/NeurIPS", "ICML", "KDD", "WWW", "HT", "WSDM", "SIGIR", "COLT", "ICDM", "CIKM", "AISTATS", "SDM", "ECML/PKDD", "ECIR", "PAKDD", "RecSys", "IJCNN", "ICANN", "ILP", "ICLR", "ACML", "ESANN", "MLJ", "JMLR", "IEEE Trans. Neural Networks", "DMKD"]),
                'do_graph_vectors': False,
                'do_semantic_vectors': False,
                'separation_year': 2016, 
                'verbose': 1
            }).get_network(), 
            'metrics': {
                'prev_co_authors': HYDRASMETRICS.OVERLAP,  # social
                'country_geocodes': HYDRASMETRICS.GEO_DISTANCE,  # geographic 
                'affi_geocodes': HYDRASMETRICS.GEO_DISTANCE,  # geographic
                # 'graph_vectors': HYDRASMETRICS.COSINE_SIMILARITY,  # social
                # 'semantic_vectors': HYDRASMETRICS.COSINE_SIMILARITY,  # social
                'citations':HYDRASMETRICS.OVERLAP,  # cognitive
                'venue':HYDRASMETRICS.OVERLAP,  # cognitive
                's2fieldsofstudy': HYDRASMETRICS.OVERLAP,  #cognitive
                'schol_affiliations': HYDRASMETRICS.OVERLAP,  # organisational
                'dblp_affiliations': HYDRASMETRICS.OVERLAP,  # organisational
                'countries': HYDRASMETRICS.OVERLAP,  # organisational
                'schol_hindex': HYDRASMETRICS.DISTANCE,  # institutional
            }, 
            'calculate_matrices': True,
            'save_matrices_path': Path("data", "matrices"),
            'calculate_evidences': True,
            'save_evidences_path': Path("data", "evidence"),
            'build_plot': False,
            'top_k_percentage': 0.01,
            'ks': [1, 3, 5, 10, 50, 100, 1000, 10000], 
            'verbose': 1
        })
