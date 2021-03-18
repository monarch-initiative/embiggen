from typing import Dict

import pandas as pd
from ensmallen_graph import EnsmallenGraph
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from extra_keras_metrics import get_standard_binary_metrics

from ..sequences import EdgePredictionDegreeSequence


class DegreePerceptron:

    def __init__(self, optimizer: str = "nadam"):
        """Create new Perceptron object."""
        self._model_name = self.__class__.__name__
        sources_input = Input((1,), name="SourceNodeDegreeInput")
        destination_input = Input((1,), name="DestinationNodeDegreeInput")
        concatenation = Concatenate()([
            sources_input,
            destination_input
        ])
        output = Dense(
            units=1,
            activation="sigmoid",
        )(concatenation)
        self._model = Model(
            inputs=[
                sources_input,
                destination_input
            ],
            outputs=output,
            name=self._model_name
        )
        self._model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=get_standard_binary_metrics(),
        )

    def fit(
        self,
        graph: EnsmallenGraph,
        batch_size: int = 2**18,
        batches_per_epoch: int = 2**10,
        negative_samples: float = 1.0,
        ** kwargs: Dict
    ) -> pd.DataFrame:
        """Train model and return training history dataframe.

        Parameters
        -------------------
        graph: EnsmallenGraph,
            Graph object to use for training.
        batch_size: int = 2**18,
            Batch size for the training process.
        batches_per_epoch: int = 2**10,
            Number of batches to train for in each epoch.
        negative_samples: float = 1.0,
            Rate of unbalancing in the batch.
        ** kwargs: Dict,
            Parameteres to pass to the dit call.

        Returns
        --------------------
        Dataframe with traininhg history.
        """
        sequence = EdgePredictionDegreeSequence(
            graph,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            negative_samples=negative_samples
        )
        return pd.DataFrame(self._model.fit(
            sequence,
            **kwargs
        ).history)

    def predict(self, *args, **kwargs):
        """Run predict."""
        return self._model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """Run predict."""
        return dict(zip(
            self._model.metrics_names,
            self._model.evaluate(*args, **kwargs)
        ))
