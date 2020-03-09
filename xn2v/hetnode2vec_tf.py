import logging.handlers
import logging
import numpy as np    # type: ignore
import os
import random
import sys
import time

from collections import defaultdict
from multiprocessing import Pool
from typing import Dict

log = logging.getLogger("xn2v.log")

handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "xn2v.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
# log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(handler)
log.addHandler(logging.StreamHandler())


class N2vGraph:
    """A class to represent perform random walks on a graph in order to derive the data needed for the node2vec
    algorithm.

    Assumptions: That the CSF graph is always undirected.

    Attributes:
        csf_graph: An undirected Compressed Storage Format graph object. Graph stores two directed edges to represent
            each undirected edge.
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

    def node2vec_walk(self, walk_length, start_node):
        """ Simulate a random walk starting from start node.

        Args:
            walk_length:
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
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    nxt = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(nxt)
            else:
                break

        return walk

    def simulate_walks(self, num_walks: int, walk_length: int):
        """Repeatedly simulate random walks from each node.

        Args:
            num_walks:
            walk_length:

        Returns:
            walks: A list of nodes, where each list constitutes a random walk.
        """

        g = self.g
        walks = []
        nodes = g.nodes_as_integers()  # this is a list
        log.info('Walk iteration:')

        for walk_iter in range(num_walks):
            print("{}/{}".format(walk_iter + 1, num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

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
        unnormalized_probs = [g.weight_from_ints(node, nbr) for nbr in g.neighbors_as_ints(node)]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

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

        # Note that g.edges returns two directed edges to represent an undirected edge between any two nodes
        # We do not need to create any additional edges for the random walk as in the Stanford implementation
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
        dsttype = dst[0]
        dst2count = defaultdict(int)  # counts for going from current node ("dst") to nodes of a given type (g, p,d)
        dst2prob = defaultdict(float)  # probs calculated from own2count
        total_neighbors = 0

        # no need to explicitly sort, g returns a sorted list
        sorted_neighbors = g.neighbors(dst)

        for nbr in sorted_neighbors:
            nbrtype = nbr[0]
            dst2count[nbrtype] += 1
            total_neighbors += 1
        total_non_own_probability = 0.0

        for n, count in dst2count.items():
            if n == dsttype:
                # we need to count up the other types before we can calculate this!
                continue
            else:
                # owntype is going to a different node type
                dst2prob[n] = float(self.gamma) / float(count)
                total_non_own_probability += dst2prob[n]

        if dst2count[dsttype] == 0:
            dst2prob[dsttype] = 0
        else:
            dst2prob[dsttype] = (1 - total_non_own_probability) / float(dst2count[dsttype])

        # Now assign the final unnormalized probs
        unnormalized_probs = np.zeros(total_neighbors)
        i = 0

        for dst_nbr in sorted_neighbors:
            nbrtype = dst_nbr[0]
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

    def __preprocess_transition_probs_xn2v(self) -> None:
        """Preprocessing of transition probabilities for guiding the random walks. This version uses gamma to
        calculate weighted skipping across a heterogeneous network.

        Assumption: The type of the node is encoded by its first character (e.g. g42 is a gene).

        Returns:
            None.
        """
        starttime = time.time()
        g = self.g
        alias_edges = {}
        alias_nodes = {}

        for node in g.nodes():
            owntype = node[0]

            # counts for going from current node ("own") to nodes of a given type(g,p,d)
            own2count: Dict = defaultdict(int)
            own2prob: Dict = defaultdict(float)  # probs calculated from own2count
            total_neighbors = 0

            # g returns a sorted list of neighbors of node
            sorted_neighbors = g.neighbors(node)

            for nbr in sorted_neighbors:
                nbrtype = nbr[0]
                own2count[nbrtype] += 1
                total_neighbors += 1

            total_non_own_probability = 0.0

            for n, count in own2count.items():
                if n == owntype:
                    # we need to count up the other types before we can calculate this!
                    continue
                else:
                    # owntype is going to a different node type
                    own2prob[n] = float(self.gamma) / float(count)
                    total_non_own_probability += own2prob[n]

            if own2count[owntype] == 0:
                own2prob[owntype] = 0
            else:
                own2prob[owntype] = (1 - total_non_own_probability) / float(own2count[owntype])

            # now assign the final unnormalized probs
            unnormalized_probs = np.zeros(total_neighbors)
            i = 0

            for nbr in sorted_neighbors:
                nbrtype = nbr[0]
                prob = own2prob[nbrtype]
                unnormalized_probs[i] = prob * g.weight(node, nbr)
                i += 1

            norm_const = sum(unnormalized_probs)
            # log.info("norm_const {}".format(norm_const))
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.__alias_setup(normalized_probs)

        for edge in g.edges():
            alias_edges[edge] = self.get_alias_edge_xn2v(edge[0], edge[1])

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
        """Compute utility lists for non-uniform sampling from discrete distributions. Refer to
        https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

        Args:
            probs: The normalized probabilities calculated by , e.g., [0.4 0.28 0.32].

        Returns:
            A list of ...
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
    def alias_draw(j, q):
        """Draw sample from a non-uniform discrete distribution using alias sampling.

        Args:

        Returns:

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

# def _repr_html_(self):
#    G = self.G
#    if isinstance(G,(nx.MultiDiGraph,nx.MultiGraph)):
#      return self.multigraph_html()

#   html = '<table><caption>Heterogeneous Node2Vec Graph Summary</caption><thead><tr><th>Node A</th>'
#  html += '<th>Node B</th><th>Edge type</th><th>Edge weight</th></tr></thead>'
#  html += "<tbody>"
#  for n,nbrs in G.adj.items():
#      for nbr,eattr in nbrs.items():
#          #wt = eattr['weight']#
#          wt = eattr.get('weight', 0)
#         et = eattr.get('edgetype','n/a')
#         html += '<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(n,nbr,et,wt)
# html += '</tbody></table>'
# return html
