import logging.handlers
import logging
import numpy as np    # type: ignore
import os
import random
import sys
import time
import tensorflow as tf  # type: ignore

from collections import defaultdict
from multiprocessing import Pool
from typing import Dict

log = logging.getLogger("xn2v.log")

handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "xn2v.log"))
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

    def node2vec_walk(self, walk_length: int, start_node) -> list:
        """ Simulate a random walk starting from start node.
        Args:
            walk_length: number of nodes to walk
            start_node:
        Returns:
            walk: A list of nodes, where each list constitutes a random walk.
        """
        g = self.g
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = g.neighbors_as_ints(cur)  # g returns a sorted list of neighbors

            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0],
                                                         alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    nxt = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0],
                                                   alias_edges[(prev, cur)][1])]
                    walk.append(nxt)
            else:
                break

        return walk

    def simulate_walks(self, num_walks: int, walk_length: int) -> tf.RaggedTensor:
        """Repeatedly simulate random walks from each node.
        Args:
            num_walks: number of individual walks to take
            walk_length: length of one walk (number of nodes)
        Returns:
            walks: A list of nodes, where each list constitutes a random walk.
        """

        g = self.g
        walks = []
        nodes = g.nodes_as_integers()  # this is a list
        log.info('Walk iteration:')

        for walk_iter in range(1, num_walks+1):
            print("{}/{}".format(walk_iter, num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length,
                                                start_node=node))

        return tf.ragged.constant(walks)

    def get_alias_edge(self, edge):
        """Get the alias edge setup lists for a given edge.

        Args:
            edge: The edge to be aliased.

        Returns:
            [original_edge, [j, q]]
        """

        src = edge[0]
        dst = edge[1]
        g = self.g
        p = self.p
        q = self.q
        unnormalized_probs = []

        for dst_nbr in g.neighbors_as_ints(dst):
            # log.info("neighbour of destination node:{}".format(dst_nbr))
            edge_weight = g.weight_from_ints(dst, dst_nbr)
            if dst_nbr == src:
                unnormalized_probs.append(edge_weight / p)
            elif g.has_edge(dst_nbr, src):
                unnormalized_probs.append(edge_weight)
            else:
                unnormalized_probs.append(edge_weight / q)

        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return [edge, self.__alias_setup(normalized_probs)]

    def _get_alias_node(self, node):
        """

        Args:
            node:

        Returns:

        """

        g = self.g
        unnormalized_probs = [g.weight_from_ints(node, nbr)
                              for nbr in g.neighbors_as_ints(node)]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const
                            for u_prob in unnormalized_probs]

        return [node, self.__alias_setup(normalized_probs)]

    def __preprocess_transition_probs(self, num_processes=8) -> None:
        """Preprocessing of transition probabilities for guiding the random walks.

        Args:
            num_processes: The number of processes to run.

        Returns:
            None.
        """
        g = self.g

        alias_nodes = {}
        num_nodes = len(g.nodes_as_integers())  # for progress updates

        with Pool(processes=num_processes) as pool:
            for i, [orig_node, alias_node] in enumerate(
                    pool.imap_unordered(self._get_alias_node, g.nodes_as_integers())):
                alias_nodes[orig_node] = alias_node
                sys.stderr.write('\rmaking alias nodes ({:03.1f}% done)'.
                                 format(100 * i / num_nodes))

        sys.stderr.write("\rDone making alias nodes.\n")

        alias_edges = {}

        # Note that g.edges returns two directed edges to represent an undirected edge
        # between any two nodes.  We do not need to create any additional edges for the
        # random walk as in the Stanford implementation
        num_edges = len(g.edges())  # for progress updates

        with Pool(processes=num_processes) as pool:
            for i, [orig_edge, alias_edge] in enumerate(
                    pool.imap_unordered(self.get_alias_edge, g.edges_as_ints())):
                alias_edges[orig_edge] = alias_edge
                sys.stderr.write('\rmaking alias edges ({:03.1f}% done)'.
                                 format(100 * i / num_edges))
        sys.stderr.write("\rDone making alias edges.\n")

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return None

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
        dst2prob = defaultdict(float)  # probs calculated from own2count
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
                # owntype is going to a different node type
                dst2prob[n] = float(self.gamma) / (float(count) * num_types)  # dividing by the num of type of edges
                total_non_own_probability += dst2prob[n]
        if dst2count[dsttype] == 0:
            dst2prob[dsttype] = 0
        else:
            dst2prob[dsttype] = (1 - total_non_own_probability) / float(dst2count[dsttype])
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
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
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
        starttime = time.time()
        G = self.g
        alias_edges = {}
        alias_nodes = {}
        for node in G.nodes():
            # ASSUMPTION. The type of the node is encoded by its first character, e.g., g42 is a gene
            # node_as_int = G.node_to_index_map[node]

            owntype = self._node_id_to_node_type(node)
            own2count: dict = defaultdict(int)  # counts for going from current node ("own") to nodes of a given type (g, p,d)
            own2prob: dict = defaultdict(float)  # probs calculated from own2count

            total_neighbors = 0
            # G returns a sorted list of neighbors of node
            sorted_neighbors = G.neighbors(node)
            for nbr in sorted_neighbors:
                nbrtype = self._node_id_to_node_type(nbr)
                own2count[nbrtype] += 1
                total_neighbors += 1
            total_non_own_probability = 0.0
            num_types = 0
            for n, count in own2count.items():  # count the number of types of edges from dst
                if own2count[n] != 0:
                    num_types += 1

            for n, count in own2count.items():
                if n == owntype:
                    # we need to count up the other types before we can calculate this!
                    continue
                else:
                    # owntype is going to a different node type
                    own2prob[n] = float(self.gamma) / float(count) * num_types
                    total_non_own_probability += own2prob[n]
            if own2count[owntype] == 0:
                own2prob[owntype] = 0
            else:
                own2prob[owntype] = (1 - total_non_own_probability) / float(own2count[owntype])
            # Now assign the final unnormalized probs
            unnormalized_probs = np.zeros(total_neighbors)
            i = 0
            for nbr in sorted_neighbors:
                nbrtype = self._node_id_to_node_type(nbr)
                prob = own2prob[nbrtype]
                unnormalized_probs[i] = prob * G.weight(node, nbr)
                i += 1
            norm_const = sum(unnormalized_probs)
            # log.info("norm_const {}".format(norm_const))
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.__alias_setup(normalized_probs)
        for edge in G.edges():
            # src_as_int = G.node_to_index_map[edge[0]]
            # dst_as_int = G.node_to_index_map[edge[1]]
            # alias_edges[(src_as_int, dst_as_int)] = self.get_alias_edge_xn2v(edge[0], edge[1])
            alias_edges[(edge[0], edge[1])] = self.get_alias_edge_xn2v(edge[0], edge[1])

        self.alias_edges = alias_edges
        self.alias_nodes = alias_nodes
        endtime = time.time()
        duration = endtime - starttime
        log.info("Setup alias probabilities for graph in {:.2f} seconds.".format(duration))
        print("Setup alias probabilities for graph in {:.2f} seconds.".format(duration))

        return None

    def retrieve_alias_nodes(self):
        """Returns alias_edges"""

        return self.alias_nodes

    def retrieve_alias_edges(self):
        """Returns alias_edges"""

        return self.alias_edges

    @staticmethod
    def __alias_setup(probs):
        """Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to:
        https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

        The alias method allows sampling from a discrete distribution in O(1) time.
        This is used here to perform biased random walks more efficiently.

        Args:
            probs: The normalized probabilities, e.g., [0.4 0.28 0.32], for
            transition to each neighbor.

        Returns:
            :param j: numpy array J for sampling from non-uniform distribution, to be
            used in alias_draw()
            :param q: numpy array q for sampling from non-uniform distribution, to be
            used in alias_draw()
        """
        k = len(probs)
        q = np.zeros(k)
        j = np.zeros(k, dtype=np.int)
        smaller = []
        larger = []

        for kk, prob in enumerate(probs):
            q[kk] = k * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            j[small] = large
            q[large] = q[large] + q[small] - 1.0

            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        return j, q

    @staticmethod
    def alias_draw(j, q) -> int:
        """Draw sample from a non-uniform discrete distribution using alias sampling.
        (See :func:`~xn2v.N2vGraph.__alias_setup`)

        Args:
        :param j: numpy array for J generated by __alias_setup
        :param q: numpy array for q generated by __alias_setup

        Returns:
        :param q: index of random sample from non-uniform discrete distribution

        """

        k = len(j)
        kk = int(np.floor(np.random.rand() * k))

        if np.random.rand() < q[kk]:
            return kk
        else:
            return j[kk]

    def __str__(self) -> str:
        """Summarizes the class for printing.

        Returns:
            'Graph'.
        """

        return 'Graph'

