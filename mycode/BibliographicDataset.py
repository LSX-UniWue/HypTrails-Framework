import networkx as nx
import numpy as np
import json

from tqdm import tqdm
from pathlib import Path

class BibliographicDataset:
    def __init__(self, args: dict) -> None:
        self.path = args['path']
        self.verbose = args['verbose']
        self.separation_year = args['separation_year']
        self.pubs = []
        with open(Path(self.path, "ai_dataset.jsonl"), 'r') as f:
            for line in tqdm(f, desc="Reading dataset ..."):
                self.pubs.append(json.loads(line))
        self.authors = dict()
        with open(Path(self.path, "persons_matched.jsonl"), 'r') as f:
            for idx, line in enumerate(tqdm(f, desc="Loading authors ...")):
                line = json.loads(line)
                line['id'] = idx
                authors = line['author'] if isinstance(line['author'], list) else [line['author']]
                for auth in authors:
                    self.authors[auth] = line
        self.geocodes = {}
        with open(Path(self.path, "country_to_coord.json"), 'r') as f:
            self.geocodes.update(json.load(f))
        with open(Path(self.path, "dblp_affiliations_coords.jsonl"), 'r') as f:
            for line in tqdm(f, desc="Loading geocodes ..."):
                line = json.loads(line)
                self.geocodes[line['affiliation']] = (line['lat'], line['lon'])
        self.node_attributes = {}
        self.graph = self.create_network()
        self.add_embedding(Path(self.path, "graph_embeddings.jsonl"), "graph")
        # self.add_embedding(Path(self.path, "semantic_author_embeddings.jsonl"), "semantic")
        nx.set_node_attributes(self.graph, self.node_attributes)
        if self.verbose:
            print(f"Having {len(self.graph)} nodes and {self.graph.size()} edges. ")
        all_keys = set([a for b in [x.keys() for x in self.node_attributes.values()] for a in b])
        print(f"Keys are: {all_keys}")

    def create_network(self) -> nx.Graph:        
        graph = nx.Graph()
        graph.add_nodes_from([author['id'] for author in self.authors.values()])
        self.node_attributes = {author['id']: {'schol_affiliations': author['schol_affiliations'] if 'schol_affiliations' in author else None, 
                                                'dblp_affiliations': author['dblp_affiliations'] if 'dblp_affiliations' in author else None,
                                                'countries': author['countries'] if 'countries' in author else None, 
                                                'country_geocodes': [self.geocodes[x] for x in author['countries'] if x in self.geocodes] if 'countries' in author else None,
                                                'affi_geocodes': [self.geocodes[x] for x in author['dblp_affiliations'] if x in self.geocodes] if 'dblp_affiliations' in author else None,
                                                'schol_hindex': author['schol_hindex'] if 'schol_hindex' in author else None, 
                                                'prev_co_authors': set(),
                                                'citations': set(),
                                                'venue': set(), 
                                                's2fieldsofstudy': set()} 
                                                for author in self.authors.values()}
        edges = []
        for pub in tqdm(self.pubs, desc="Creating edges ..."):
            auth = [self.authors[x]['id'] for x in pub['author'] if x in self.authors]
            if int(pub['year']) <= self.separation_year:
                for an_auth in auth:
                    self.node_attributes[an_auth]['prev_co_authors'].update(auth)
                    self.node_attributes[an_auth]['venue'].add(pub['venue'])
                    if 'citations' in pub and pub['citations'] is not None:
                        self.node_attributes[an_auth]['citations'].update(pub['citations'])
                    if 's2fieldsofstudy' in pub and pub['s2fieldsofstudy'] is not None:
                       self.node_attributes[an_auth]['s2fieldsofstudy'].update(set([x['category'] for x in pub['s2fieldsofstudy']]))
            else:    
                edges.extend([(auth[k], auth[v], {'year': pub['year'], 'venue': pub['venue']}) for k in range(len(auth)) for v in range(len(auth)) if k != v])
        graph.add_edges_from(edges)
        return graph
    
    def add_embedding(self, path_to_embeddings: Path, name: str) -> None:
        with open(path_to_embeddings, 'r') as f:
            for line in tqdm(f, desc="Load graph embeddings ..."):
                line = json.loads(line)
                author_id = self.authors[line['author']]['id']
                self.node_attributes[author_id][f"{name}_vectors"] = np.array(line['embedding'])

    def get_network(self) -> nx.DiGraph:
        return self.graph
    


if __name__ == "__main__":
    graph = BibliographicDataset(args={
        "path": Path("data", "bibliometric_dataset"),
        "separation_year": 2016, 
        'verbose': 1
    }).get_network()
