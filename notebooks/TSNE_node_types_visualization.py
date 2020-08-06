import os
from glob import glob
from tqdm.auto import tqdm
import numpy as np
from ensmallen_graph import EnsmallenGraph
from itertools import product

#try:
#    from tsnecuda import TSNE
#except ModuleNotFoundError:
from MulticoreTSNE import MulticoreTSNE as TSNE


from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS, ListedColormap

import matplotlib.cm as cm



embedding_path = "./FOURTH/SkipGram_embedding.npy"

graph = EnsmallenGraph.from_csv(
    edge_path="/global/scratch/marcin/N2V/MicrobeEnvironmentGraphLearn/ENIGMA_data/masterG.edgelist_col12_head.tsv",
    sources_column="subject",
    destinations_column="object",
    directed=False
)



perplexities = (20)#, 50, 80, 200, 500)

#os.makedirs("tsne", exist_ok=True)

for perplexity in tqdm(perplexities, desc="Perplexities", leave=False):
    tsne_path = f"tnse_microbeenv/microbeenv_node_{perplexity}"
    if os.path.exists(tsne_path):
        continue
    tsne = TSNE(
        perplexity=perplexity
    )
    np.save(
        tsne_path,
        tsne.fit_transform(np.load(embedding_path))
    )


node_types_columns = ("type")

def load_graph(node_type_column: str, default_node_type: str = "unknown"):
    """Load the graph with the specified node type column.

    Parameters
    ------------------------
    node_type_column:str,
        The column to be loaded.
    default_node_type:str="unknown",
        The default value to use when no node type is available for a given node.
    """
    return EnsmallenGraph.from_csv(
        edge_path="/global/scratch/marcin/N2V/MicrobeEnvironmentGraphLearn/ENIGMA_data/masterG.edgelist_col12_head.tsv",
        sources_column="subject",
        destinations_column="object",
        directed=False,
        #weights_column="weight",
        node_path="/global/scratch/marcin/N2V/MicrobeEnvironmentGraphLearn/ENIGMA_data/masterG.edgelist_col12_nodes_meta_header.txt",
        nodes_column="id",
        node_types_column=node_type_column,
        default_node_type=default_node_type
    )




# TODO: Move this method into embiggen
def plot_embedding(
        graph: EnsmallenGraph,
        tsne_embedding: np.ndarray,
        k: int = 10,
        axes: Axes = None
):
    if axes is None:
        _, axes = plt.subplots(figsize=(5, 5))

    if graph.node_types_mapping is None:
        node_types = np.zeros(graph.get_nodes_number(), dtype=np.uint8)
        common_node_types_names = ["No node type provided"]
    else:
        nodes, node_types = graph.get_top_k_nodes_by_node_type(k)
        tsne_embedding = tsne_embedding[nodes]
        common_node_types_names = list(np.array(graph.node_types_reverse_mapping)[np.unique(node_types)])

    colors = list(TABLEAU_COLORS.keys())[:len(common_node_types_names)]

    scatter = axes.scatter(
        *tsne_embedding.T,
        s=0.25,
        c=node_types,
        cmap=ListedColormap(colors)
    )
    axes.legend(
        handles=scatter.legend_elements()[0],
        labels=common_node_types_names
    )
    return axes



def plot_embedding_degrees_heatmap(
    graph:EnsmallenGraph,
    embedding:np.ndarray,
    axes:Axes=None,
    fig:Figure=None,
    max_degree=10
):
    if axes is None:
        fig, axes = plt.subplots(figsize=(10,10), dpi=200)

    cm = plt.cm.get_cmap('RdYlBu')
    degrees = graph.degrees()
    degrees[degrees > max_degree] = min(max_degree, degrees.max())
    sc = axes.scatter(*embedding.T, c=degrees, s=0.1, cmap=cm)
    fig.colorbar(sc, ax=axes)


fig, axes_matrix = plt.subplots(
    ncols=len(perplexities),
    nrows=len(node_types_columns),
    figsize=(len(perplexities)*6, len(node_types_columns)*6),
    dpi=150
)
for node_types_column, axes_row in zip(node_types_columns, axes_matrix):
    for perplexity, axes in zip(perplexities, axes_row):
        tsne_embedding = np.load(f"tnse_microbeenv/microbeenv_node_{perplexity}")
        plot_embedding(
            load_graph(node_types_column),
            tsne_embedding,
            axes=axes
        )
        axes.set_title(f"Perp. {perplexity}")
###SAVE
plt.savefig('microbeenv_node_types.png')
#plt.show()


graph = load_graph(node_types_columns[0])

fig, axes_row = plt.subplots(
    ncols=len(perplexities),
    figsize=(len(perplexities)*8, 6),
    dpi=150
)
for perplexity, axes in zip(perplexities, axes_row):
    tsne_embedding = np.load(f"tnse_microbeenv/microbeenv_node_{perplexity}")
    plot_embedding_degrees_heatmap(
        graph,
        tsne_embedding,
        axes=axes,
        fig=fig
    )
    axes.set_title(f"Perp. {perplexity}")
###SAVE
plt.savefig('micvobe_env_degrees.png')
#plt.show()