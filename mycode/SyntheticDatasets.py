import networkx as nx
import numpy as np

UNIVERSITIES = ['Heidelberg University', 'Leipzig University', 'University of Rostock', 'University of Greifswald', 'University of Freiburg', 'University of Munich', 'University of Tübingen', 'University of Halle-Wittenberg', 'University of Marburg', 'University of Jena']

COORDS = {'lat': (47.5, 53.77), 'long': (7.5, 14.1)}


class SyntheticDataset:
    def __init__(self, args: dict) -> None:
        self.args = args
        self.graph = nx.generators.barabasi_albert_graph(n=args['size'], m=args['connections'])
        attribute_dict = self.generate_random_attributes(args['attributes'])
        nx.set_node_attributes(self.graph, attribute_dict)

    def generate_random_attributes(self, attributes: dict) -> dict:
        ret_dict = {x: {} for x in range(self.args['size'])}
        for a_name, a_type in attributes.items():
            for k in ret_dict.keys():
                if a_name == "affiliations":
                    values = [x[0](*(x[1], x[2](*x[3]))) for x in a_type]
                elif a_name == "geo_coords":
                    values = [[x[0](*x[1:]) for x in a_type[:-1]] for _ in range(a_type[-1])]
                else:
                    values = [x[0](*x[1:]) for x in a_type]
                if len(values) == 1:
                    values = values[0]
                if isinstance(values, np.ndarray):
                    values = values.tolist()
                if a_name == "affiliations":
                    values = set(values)
                ret_dict[k].update({a_name: values})
        return ret_dict


    def get_network(self) -> nx.DiGraph:
        return self.graph

if __name__ == "__main__":
    graph = SyntheticDataset(args={
        'size': 20,
        'connections': 2,
        'attributes': {
            'age': [(np.random.choice, np.arange(18,65))],  # rand int between
            'gender': [(np.random.choice, ['m', 'f'])],  # either one or another
            'affiliations': [(np.random.choice, UNIVERSITIES, np.random.randint, (1, 5))],  # one to 5 affiliations todo fix that it generates individually
            'geo_coords': [(np.random.uniform, *COORDS['lat']), (np.random.uniform, *COORDS['long']), np.random.randint(1, 5)],  # generate random geo coords within germany
            'semantic_representation':  [(np.random.rand, 768)]  # create a random vector representation
        } 
    }).get_network()
    for node in graph.nodes():
        print(node)