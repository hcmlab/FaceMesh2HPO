from typing import List, Tuple

import networkx as nx
from loguru import logger

from src.hpo_tree.hpo_loader import load_hpo
from src.hpo_tree.hpo_term import HumanPhenotypeTerm


def hpo_graph(output_dir: str, root_node: str = 'HP:0000271',
              target_nodes: List[str] = ['HP:0000478', 'HP:0000271'], download: bool = True):
    hpo = load_hpo(output_dir, download)
    G = nx.DiGraph()
    for term in hpo.terms():
        if not term.obsolete:
            G.add_node(term.id, name=term.name)
            for parent in term.superclasses(distance=1):
                if not parent.obsolete:
                    G.add_edge(parent.id, term.id)

    dependent_nodes = set()
    for target_node in target_nodes:
        if target_node in G:
            dependent_nodes.update(nx.descendants(G, target_node))

    dependent_nodes.update(target_nodes)
    graph = G.subgraph(dependent_nodes).copy()
    roots = [n for n, d in graph.in_degree() if d == 1]
    for root in roots:
        incoming_edges = list(graph.in_edges(root))
        graph.remove_edges_from(incoming_edges)

    for target_node in target_nodes:
        if target_node != root_node:
            graph.add_edge(root_node, target_node)

    def recursive_edge_accumulation(node_name, edges: List[Tuple[str, str]] = [], children: List[str] = []):
        for child in graph.successors(node_name):
            if child != node_name and child not in children:
                edges.append((node_name, child))
                children.append(child)
                recursive_edge_accumulation(child, edges, children)
        return edges

    edges = recursive_edge_accumulation(root_node)
    edges_to_remove = [edge for edge in graph.edges if edge not in edges]
    for edge in edges_to_remove:
        graph.remove_edge(edge[0], edge[1])

    logger.debug(
        f"Reduced graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges saved under {graph_file}.")
    return graph


def build_modified_hpo_tree(data_dir: str, download: bool = False):
    hpo = HumanPhenotypeTerm.load_ontology(data_dir, download=download)
    hp_abnorm_face = hpo.find_successor('HP:0000271')  # Abnormality of the face
    hp_abnorm_eye = hpo.find_successor('HP:0000478')  # Abnormality of the eye
    hp_abnorm_eyebrow = hpo.find_successor('HP:0000534')  # Abnormal eyebrow morphology
    hp_abnorm_face.add_successor(hp_abnorm_eye)  # Move eye to face
    hp_abnorm_face.add_successor(hp_abnorm_eyebrow)  # Move eyebrow to face
    hp_abnorm_face.define_as_root()
    return hp_abnorm_face
