import silence_tensorflow.auto

from ensmallen_graph import EnsmallenGraph

graph = EnsmallenGraph.from_csv(
    edge_path="/global/scratch/marcin/N2V/MicrobeEnvironmentGraphLearn/ENIGMA_data/masterG.edgelist_col12_head.tsv",
    sources_column="subject",
    destinations_column="object",
    directed=False
    #weights_column="weight"
)

print(graph.report())

training, validation = graph.connected_holdout(42, 0.8)

assert graph > training
assert graph > validation
assert (training + validation).contains(graph)
assert graph.contains(training + validation)
assert not training.overlaps(validation)
assert not validation.overlaps(training)

walk_length=50
batch_size=2**8
iterations=20
window_size=4
p=1.0
q=1.0
embedding_size=100
negatives_samples=30
patience=2
delta=0.0001
epochs=10
learning_rate=0.1

from embiggen import Node2VecSequence

training_sequence = Node2VecSequence(
    training,
    walk_length=walk_length,
    batch_size=batch_size,
    iterations=iterations,
    window_size=window_size,
    return_weight=1/p,
    explore_weight=1/q
)

validation_sequence = Node2VecSequence(
    graph, # Here we use the entire graph. This will only be used for the early stopping.
    walk_length=walk_length,
    batch_size=batch_size,
    iterations=iterations,
    window_size=window_size,
    return_weight=1/p,
    explore_weight=1/q
)

#CREATING
#from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.optimizers import Nadam
from embiggen import SkipGram

#strategy = MirroredStrategy()
#with strategy.scope():
model = SkipGram(
       	vocabulary_size=training.get_nodes_number(),
	embedding_size=embedding_size,
       	window_size=window_size,
      	negatives_samples=negatives_samples,
        optimizer=Nadam(learning_rate=learning_rate)
)

print(model.summary())

#TUNING
from tensorflow.keras.callbacks import EarlyStopping

history = model.fit(
    training_sequence,
    steps_per_epoch=training_sequence.steps_per_epoch,
    validation_data=validation_sequence,
    validation_steps=validation_sequence.steps_per_epoch,
    epochs=epochs,
    callbacks=[
        EarlyStopping(
            "val_loss",
            min_delta=delta,
            patience=patience,
            restore_best_weights=True
        )
    ]
)

#SAVE
model.save_weights(f"{model.name}_weights.h5")

import numpy as np

np.save(f"{model.name}_embedding.npy", model.embedding)


#from plot_keras_history import plot_history

#plot_history(history)
