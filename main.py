import time
import joblib
import numpy as np
import networkx as nx
from pathlib import Path
from functools import partial
from scipy.sparse import csr_matrix, vstack
from pytrails.hyptrails import MarkovChain

from mycode.Metrics import *
from mycode.SyntheticDatasets import *
from mycode.BibliographicDataset import *
from mycode.Visualization import plot_hyptrails

def calculate_row_similarity(row, attributes: dict, metric: HYDRASMETRIC, top_k: int) -> csr_matrix:
    current = attributes[row]
    if current is None or (isinstance(current, list) and len(current) == 0):
        return row, csr_matrix((1, len(attributes.keys())))
    if top_k:
        curr = [(idy, metric.get_similarity(current, val)) for idy, val in attributes.items()]
        curr = sorted(curr, key=lambda x: x[1], reverse=True)[:top_k]
        return_csr = csr_matrix((1, len(attributes.keys())))
        for col, sim in curr:
            return_csr[0, col] = sim
        return row, return_csr
    curr = [metric.get_similarity(current, val) for val in attributes.values()]
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
            self.hypotheses = {}
            for file in Path(self.save_matrices_path).iterdir():
                if file.name.startswith(self.run_name):
                    self.hypotheses[file.name.split("-")[1].split(".")[0]] = np.load(file, allow_pickle=True).item()
            self.transitions = np.load(Path(self.save_matrices_path, f"{self.run_name}-transitions.npy"), allow_pickle=True).item()
        if self.calculate_evidences:
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
        hypotheses = {x.name: csr_matrix(self.transitions.shape) for x in self.metrics}
        for metric in self.metrics:
            print(f"\tCalculating {metric.name} ...")
            attributes = {self.nodes_to_id[node]: self.network.nodes[node][metric.name] for node in self.network.nodes()}
            partial_similarity = partial(calculate_row_similarity, metric=metric, attributes=attributes, top_k=self.top_k)
            tmp_hypotheses = {}
            start = time.time()
            with joblib.Parallel(n_jobs=WORKER, verbose=10) as parallel:  
                for row, sim_row in parallel(joblib.delayed(partial_similarity)(i) for i in range(len(self.network.nodes()))):
                    tmp_hypotheses[row] = sim_row
            whole = vstack(tmp_hypotheses[i] for i in range(len(self.network.nodes())))
            hypotheses[metric.name] = whole
            if metric.metric == HYDRASMETRICDISTANCE.GEO_DISTANCE or metric.metric == HYDRASMETRICDISTANCE.DISTANCE:
                curr_max = hypotheses[metric.name].max()
                new_scores = 1 - hypotheses[metric.name].data / curr_max
                hypotheses[metric.name].data = new_scores
            hypotheses[metric.name] = self.save_normalize(hypotheses[metric.name])
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
            'run_name': 'synthetic',
            'network': SyntheticDataset(args={
                'size': 100,
                'connections': 5,
                'attributes': {
                    'age': [(np.random.choice, np.arange(18,65))],  # rand int between
                    'gender': [(np.random.choice, ['m', 'f'])],  # either one or another
                    'affiliations': [(np.random.choice, UNIVERSITIES, np.random.randint, (1, 5))],  # one to 5 affiliations todo fix that it generates individually
                    'geo_coords': [(np.random.uniform, *COORDS['lat']), (np.random.uniform, *COORDS['long']), np.random.randint(1, 5)],  # generate random geo coords within germany
                    'semantic_representation': [(np.random.rand, 768)]  # create a random vector representation
                } 
            }).get_network(),
            'metrics': [
                HYDRASMETRIC(name="age", metric=HYDRASMETRICDISTANCE.DISTANCE), 
                HYDRASMETRIC(name="gender", metric=HYDRASMETRICDISTANCE.EQUALS), 
                HYDRASMETRIC(name="affiliations", metric=HYDRASMETRICDISTANCE.OVERLAP), 
                HYDRASMETRIC(name="geo_coords", metric=HYDRASMETRICDISTANCE.GEO_DISTANCE), 
                HYDRASMETRIC(name="semantic_representation", metric=HYDRASMETRICDISTANCE.COSINE_SIMILARITY)
            ],
            'calculate_matrices': False,
            'save_matrices_path': Path("data", "matrices"),
            'calculate_evidences': True,
            'save_evidences_path': Path("data", "evidence"),
            'build_plot': True,
            'top_k_percentage': 0.01,
            'ks': [1, 3, 5, 10, 50, 100, 1000, 10000], 
            'verbose': 1
        })
    else:
        ai_confs = set(["NIPS/NeurIPS", "ICML", "KDD", "WWW", "HT", "WSDM", "SIGIR", "COLT", "ICDM", "CIKM", "AISTATS", "SDM", "ECML/PKDD", "ECIR", "PAKDD", "RecSys", "IJCNN", "ICANN", "ILP", "ICLR", "ACML", "ESANN", "MLJ", "JMLR", "IEEE Trans. Neural Networks", "DMKD"])
        Hydras(args={
            'run_name': 'nips-vectors',
            'network': BibliographicDataset(args={
                'path': Path("data", "bibliometric_dataset"),
                'conferences': ["NIPS/NeurIPS"],  # None would mean all
                'do_graph_vectors': True,
                'do_semantic_vectors': True,
                'separation_year': 2016, 
                'verbose': 1
            }).get_network(), 
            'metrics': [
                HYDRASMETRIC(name="prev_co_authors", metric=HYDRASMETRICDISTANCE.OVERLAP),  # social
                HYDRASMETRIC(name="graph_vectors", metric=HYDRASMETRICDISTANCE.COSINE_SIMILARITY),  # social
                HYDRASMETRIC(name="semantic_vectors", metric=HYDRASMETRICDISTANCE.COSINE_SIMILARITY),  # social
                HYDRASMETRIC(name="country_geocodes", metric=HYDRASMETRICDISTANCE.GEO_DISTANCE),  # geographic
                HYDRASMETRIC(name="affi_geocodes", metric=HYDRASMETRICDISTANCE.GEO_DISTANCE),  # geographic
                HYDRASMETRIC(name="citations", metric=HYDRASMETRICDISTANCE.OVERLAP),  # cognitive
                HYDRASMETRIC(name="venue", metric=HYDRASMETRICDISTANCE.OVERLAP),  # cognitive
                HYDRASMETRIC(name="s2fieldsofstudy", metric=HYDRASMETRICDISTANCE.OVERLAP),  # cognitive
                HYDRASMETRIC(name="schol_affiliations", metric=HYDRASMETRICDISTANCE.OVERLAP),  # organisational
                HYDRASMETRIC(name="dblp_affiliations", metric=HYDRASMETRICDISTANCE.OVERLAP),  # organisational
                HYDRASMETRIC(name="countries", metric=HYDRASMETRICDISTANCE.OVERLAP),  # organisational
                HYDRASMETRIC(name="schol_hindex", metric=HYDRASMETRICDISTANCE.DISTANCE),  # institutional
            ], 
            'calculate_matrices': True,
            'save_matrices_path': Path("data", "matrices"),
            'calculate_evidences': True,
            'save_evidences_path': Path("data", "evidence"),
            'build_plot': False,
            'top_k_percentage': 0.01,
            'ks': [1, 3, 5, 10, 50, 100, 1000, 10000], 
            'verbose': 1
        })
