import os
from glob import glob
from tqdm.auto import tqdm
import numpy as np
from ensmallen_graph import EnsmallenGraph
from embiggen import GraphTransformer, EdgeTransformer

#try:
#    from tsnecuda import TSNE
#except ModuleNotFoundError:
from MulticoreTSNE import MulticoreTSNE as TSNE

embedding_path = "./FOURTH/SkipGram_embedding.npy"

graph = EnsmallenGraph.from_csv(
    edge_path="/global/scratch/marcin/N2V/MicrobeEnvironmentGraphLearn/ENIGMA_data/masterG.edgelist_col12_head.tsv",
    sources_column="subject",
    destinations_column="object",
    directed=False
)


negative_graph = graph.sample_negatives(42, graph.get_edges_number(), False)

embedding = np.load(embedding_path)

for method in tqdm(EdgeTransformer.methods, desc="Methods", leave=False):
    tsne_path = f"tsne_edges_microbeenv"
    if os.path.exists(tsne_path):
        continue
    transformer = GraphTransformer(method)
    transformer.fit(embedding)
    positive_edges = transformer.transform(graph)
    negative_edges = transformer.transform(negative_graph)
    edges = np.vstack([positive_edges, negative_edges])
    nodes = np.concatenate([
        np.ones(positive_edges.shape[0]),
        np.zeros(negative_edges.shape[0])
    ])
    indices = np.arange(0, nodes.size)
    np.random.shuffle(indices)
    edges = edges[indices]
    nodes = nodes[indices]
    np.save(f"tsne_edges_microbeenv_labels", nodes)
    tsne = TSNE(verbose=True)
    np.save(
        tsne_path,
        tsne.fit_transform(edges)
    )