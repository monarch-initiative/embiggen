from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from .test_link_prediction_sequence import TestLinkPredictionSequence


class TestLinkPredictionModel(TestLinkPredictionSequence):

    def setUp(self):
        super().setUp()
        self._model = Sequential([
            Dense(1, activation="relu")
        ])
        self._model.compile(loss="binary_crossentropy")

    def test_fit(self):
        self._model.fit(
            self._sequence,
            steps_per_epoch=self._sequence.steps_per_epoch,
            epochs=2,
            verbose=False
        )
