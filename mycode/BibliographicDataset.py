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
        self.do_graph_vectors = args['do_graph_vectors']
        self.do_semantic_vectors = args['do_semantic_vectors']
        if 'conferences' not in args or args['conferences'] is None:
            with open(Path(self.path, "ai_venues.json"), 'r') as f:
                tmp = json.load(f)
                self.conferences = set([a for b in tmp.values() for a in b])
        else:
            self.conferences = args['conferences']
        self.pubs = []
        self.authors_to_consider = set()
        with open(Path(self.path, "ai_dataset.jsonl"), 'r') as f:
            for line in tqdm(f, desc="Reading dataset ..."):
                line = json.loads(line)
                if line['venue'] in self.conferences:
                    self.pubs.append(line)
                    self.authors_to_consider.update(line['author'])
        print(f"Having {len(self.pubs)} publications and {len(self.authors_to_consider)} authors. ")
        self.authors = dict()
        with open(Path(self.path, "persons_matched.jsonl"), 'r') as f:
            for line in tqdm(f, desc="Loading authors ..."):
                line = json.loads(line)
                authors = line['author'] if isinstance(line['author'], list) else [line['author']]
                for auth in authors:
                    if auth in self.authors_to_consider:
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
        if self.do_graph_vectors:
            self.add_embedding(Path(self.path, "graph_embeddings.jsonl"), "graph")
        if self.do_semantic_vectors:
            self.add_embedding(Path(self.path, "semantic_author_embeddings.jsonl"), "semantic")
        nx.set_node_attributes(self.graph, self.node_attributes)
        if self.verbose:
            print(f"Having {len(self.graph)} nodes and {self.graph.size()} edges. ")
        all_keys = set([a for b in [x.keys() for x in self.node_attributes.values()] for a in b])
        print(f"Keys are: {all_keys}")

    def create_network(self) -> nx.Graph:        
        graph = nx.Graph()
        graph.add_nodes_from([author['id'] for author in self.authors.values()])
        self.node_attributes = {author['id']: {'schol_affiliations': set(author['schol_affiliations']) if 'schol_affiliations' in author and author['schol_affiliations'] is not None else None, 
                                                'dblp_affiliations': set(author['dblp_affiliations']) if 'dblp_affiliations' in author and author['dblp_affiliations'] is not None else None,
                                                'countries': set(author['countries']) if 'countries' in author and author['countries'] is not None else None, 
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
                if line['author'] not in self.authors:
                    continue
                author_id = self.authors[line['author']]['id']
                self.node_attributes[author_id][f"{name}_vectors"] = np.array(line['embedding'])

    def get_network(self) -> nx.DiGraph:
        return self.graph
    


if __name__ == "__main__":
    graph = BibliographicDataset(args={
        'path': Path("data", "bibliometric_dataset"),
        'conferences': set(["NIPS/NeurIPS", "ICML", "KDD", "WWW"]),
        "separation_year": 2016, 
        'verbose': 1
    }).get_network()
