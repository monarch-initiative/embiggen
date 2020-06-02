import logging.handlers
import logging
import numpy as np    # type: ignore
import os
import random
import sys
import time
import tensorflow as tf  # type: ignore
from tqdm.auto import trange  # type: ignore

from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, Tuple

log = logging.getLogger("embiggen.log")

handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "embiggen.log"))
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
# log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(handler)
log.addHandler(logging.StreamHandler())


class N2vGraph:
    """A class to represent perform random walks on a graph in order to derive the data
    needed for the node2vec algorithm.

    Assumptions: That the CSF graph is always undirected.

    Attributes:
        csf_graph: An undirected Compressed Storage Format graph object. Graph stores
        two directed edges to represent each undirected edge.
        p:
        q:
        gamma:
        doxn2v:
    """

    def __init__(self, csf_graph, p, q, gamma, doxn2v=True) -> None:

        self.g = csf_graph
        self.p = p
        self.q = q
        self.gamma = gamma

        if doxn2v:
            self.__preprocess_transition_probs_xn2v()
        else:
            self.__preprocess_transition_probs()

    def get_alias_edge_xn2v(self, src, dst):
        """Gets the alias edge setup lists for a given edge.

        Args:
            src:
            dst:

        Returns:

        """
        g = self.g
        p = self.p
        q = self.q
        # log.info("source node:{}".format(src))
        # log.info("destination node:{}".format(dst))
        dsttype = self._node_id_to_node_type(dst)
        # counts for going from current node ("dst") to nodes of a given type (g, p,d)
        dst2count = defaultdict(int)
        dst2prob = defaultdict(float)  # probs calculated from neighbours_count_per_types
        total_neighbors = 0

        # No need to explicitly sort, g returns a sorted list
        sorted_neighbors = g.neighbors(dst)
        for nbr in sorted_neighbors:
            nbrtype = self._node_id_to_node_type(nbr)
            dst2count[nbrtype] += 1
            total_neighbors += 1


        total_non_own_probability = 0.0
        num_types = 0
        for n, count in dst2count.items():  # count the number of types of edges from dst
            if dst2count[n] != 0:
                num_types += 1

        for n, count in dst2count.items():
            if n == dsttype:
                # we need to count up the other types before we can calculate this!
                continue
            else:
                # node_type is going to a different node type
                # dividing by the num of type of edges
                dst2prob[n] = float(self.gamma)  / (float(count) * num_types)

                total_non_own_probability += dst2prob[n]

                
        if dst2count[dsttype] == 0:
            dst2prob[dsttype] = 0
        else:
            dst2prob[dsttype] = (
                1 - total_non_own_probability) / float(dst2count[dsttype])


        # Now assign the final unnormalized probs
        unnormalized_probs = np.zeros(total_neighbors)
        i = 0
        for dst_nbr in sorted_neighbors:
            nbrtype = self._node_id_to_node_type(dst_nbr)
            prob = dst2prob[nbrtype]
            edge_weight = g.weight(dst, dst_nbr)
            if dst_nbr == src:
                unnormalized_probs[i] = prob * edge_weight / p
            elif g.has_edge(dst_nbr, src):
                unnormalized_probs[i] = prob * edge_weight
            else:
                unnormalized_probs[i] = prob * edge_weight / q
            i += 1
    
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob) / norm_const for u_prob in unnormalized_probs]
            
        return self.__alias_setup(normalized_probs)

    def _node_id_to_node_type(self, id: str) -> str:
        """Given a node id, return node type using self.g.index_to_nodetype_map

        :param id: node id
        :return: node type
        """
        return self.g.index_to_nodetype_map[self.g.node_to_index_map[id]]

    def __preprocess_transition_probs_xn2v(self) -> None:
        """Preprocessing of transition probabilities for guiding the random walks. This version uses gamma to
        calculate weighted skipping across a heterogeneous network.

        Assumption: The type of the node is encoded by its first character (e.g. g42 is a gene).

        Returns:
            None.
        """
        G = self.g
        alias_edges = {}
        alias_nodes = {}
        for node in G.nodes():
            # ASSUMPTION. The type of the node is encoded by its first character, e.g., g42 is a gene
            # node_as_int = G.node_to_index_map[node]

            node_type = self._node_id_to_node_type(node)
            # counts for going from current node ("own") to nodes of a given type (g, p,d)
            neighbours_count_per_types: dict = defaultdict(int)
            # probs calculated from neighbours_count_per_types
            # for each type of neighbours the probability of jumping to it
            # eg
            # {
            #   "type_of_node_a":0.33,
            #   "type_of_node_b":0.66
            # }
            neighbours_count_per_types_prob: dict = defaultdict(float)

            total_neighbors = 0
            # G returns a sorted list of neighbors of node
            sorted_neighbors = G.neighbors(node)
            # For each neighbour
            for nbr in sorted_neighbors:
                # Get the node type
                nbrtype = self._node_id_to_node_type(nbr)
                # And get the type of the node
                neighbours_count_per_types[nbrtype] += 1
                # And increase the total number of neighbour
                total_neighbors += 1
            
            total_non_own_probability = 0.0
            num_types = len(neighbours_count_per_types)

            
            for neighbour_type, count in neighbours_count_per_types.items():
                if neighbour_type == node_type:
                    # we need to count up the other types before we can calculate this!
                    continue
                else:
                    # node_type is going to a different node type
                    # TODO: use TOMMY version of GAMMA application
                    # TODO: only use GAMMA for the edges, as it is used in the current implementation
                    neighbours_count_per_types_prob[neighbour_type] = float(self.gamma) / float(count) * num_types
                    total_non_own_probability += neighbours_count_per_types_prob[neighbour_type]

            if neighbours_count_per_types[node_type] == 0:
                neighbours_count_per_types_prob[node_type] = 0
            else:
                # Probability of remaining type
                neighbours_count_per_types_prob[node_type] = (
                    1 - total_non_own_probability) / float(neighbours_count_per_types[node_type])
            # Now assign the final unnormalized probs
            unnormalized_probs = np.zeros(total_neighbors)
            i = 0
            for nbr in sorted_neighbors:
                nbrtype = self._node_id_to_node_type(nbr)
                prob = neighbours_count_per_types_prob[nbrtype]
                unnormalized_probs[i] = prob * G.weight(node, nbr)
                i += 1
            norm_const = sum(unnormalized_probs)
            # log.info("norm_const {}".format(norm_const))
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.__alias_setup(normalized_probs)
        for edge in G.edges():
            # src_as_int = G.node_to_index_map[edge[0]]
            # dst_as_int = G.node_to_index_map[edge[1]]
            # alias_edges[(src_as_int, dst_as_int)] = self.get_alias_edge_xn2v(edge[0], edge[1])
            alias_edges[edge] = self.get_alias_edge_xn2v(*edge)

        self.alias_edges = alias_edges
        self.alias_nodes = alias_nodes

        return None