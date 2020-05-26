from tensorflow.keras.models import Sequential, Model  # type: ignore  # pylint: disable=import-error
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, Concatenate, Layer  # type: ignore  # pylint: disable=import-error
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore  # pylint: disable=import-error
from tensorflow.keras.metrics import AUC  # type: ignore  # pylint: disable=import-error
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Tuple, Dict, List, Union, Optional


class NeuralNetwork:
    def __init__(
        self,
        max_epochs: int = 1000,
        batch_size: int = 64,
        monitor: str = "auprc",
        patience: int = 10,
    ):
        """Instantiate a new NeuralNetwork.

            Parameters
            ----------------------
            max_epochs: int = 1000,
                Maximum number of epochs for which to train the model.
                It can be interrupted early by the early stopping.
            batch_size: int = 64,
                Number of samples to take in consideration for each batch.
            monitor: str = "auprc",
                Metric to monitor for the early stopping.
                It is possible to use validation metrics when training the model
                in contexts such as gaussian processes, where inner holdouts are used.
                Such metrics could be "val_auroc", "val_auprc" or "val_loss".
                Using validation metrics in non-inner holdouts is discouraged
                as it can be seen as epochs overfitting for the test set.
                When training the model on non-inner holdouts, use metrics
                such as "auroc", "auprc" or "loss".
            patience: int = 10,
                Number of epochs to wait for an improvement.
        """
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._monitor = monitor
        self._patience = patience
        self._model = self._build_model()
        self._compile_model()

    def _build_model(self) -> Model:
        raise NotImplementedError(
            "The method _build_model has to be implemented in the subclasses."
        )

    def _compile_model(self) -> Model:
        self._model.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(curve="ROC", name="auroc"),
                AUC(curve="PR", name="auprc")
            ]
        )

    def predict_proba(self, *args, **kwargs):
        predictions = self._model.predict(*args, **kwargs)
        return np.hstack([
            1-predictions,
            predictions
        ])

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs).round().astype(int)

    def fit(
        self,
        train_x: Union[Dict, List, np.ndarray],
        train_y: Union[Dict, List, np.ndarray],
        test_x: Union[Dict, List, np.ndarray] = None,
        test_y: Union[Dict, List, np.ndarray] = None
    ) -> pd.DataFrame:
        """Fit the model using given training parameters.

        Parameters
        --------------------------
        train: Tuple[Union[Dict, List, np.ndarray]],
            Either a tuple of list, np.ndarrays or dictionaries,
            containing the training data.
        test: Tuple[Union[Dict, List, np.ndarray]] = None
            Either a tuple of list, np.ndarrays or dictionaries,
            containing the validation data data.
            These data are optional, but they are required if
            the given monitor metric starts with "val_"

        Raises
        --------------------------
        ValueError,
            If no test data are given but the monitor metric
            starts with the word "val_", meaning it has to be
            computed on the test data.

        Returns
        ---------------------------
        The training history as a pandas dataframe.
        """
        test = None
        if all(d is not None for d in (test_x, test_y)):
            test = (test_x, test_y)
        if test is None and self._monitor.startswith("val_"):
            raise ValueError(
                "No test set was given, "
                "but a validation metric was required for the early stopping."
            )
        return pd.DataFrame(self._model.fit(
            train_x, train_y,
            epochs=self._max_epochs,
            batch_size=self._batch_size,
            validation_data=test,
            verbose=False,
            shuffle=True,
            callbacks=[
                EarlyStopping(self._monitor, patience=self._patience),
            ]
        ).history)


class MLP(NeuralNetwork):

    def __init__(self, input_shape: Tuple, *args, **kwargs):
        self._input_shape = input_shape
        super().__init__(*args, **kwargs)

    def _build_model(self) -> Model:
        return Sequential([
            Input(self._input_shape),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ], name="MLP")


class FFNN(NeuralNetwork):

    def __init__(self, input_shape: Tuple, *args, **kwargs):
        self._input_shape = input_shape
        super().__init__(*args, **kwargs)

    def _build_model(self) -> Model:
        return Sequential([
            Input(self._input_shape),
            Dense(128, activation="relu"),
            Dense(128),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1, activation="sigmoid"),
        ], name="FFNN")


class MultiModalFFNN(NeuralNetwork):

    def __init__(self, input_shape: Tuple, *args, **kwargs):
        self._input_shape = input_shape
        super().__init__(*args, **kwargs)

    def _sub_module(self, previous: Layer) -> Layer:
        hidden = Dense(128, activation="relu")(previous)
        hidden = Dense(128)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation("relu")(hidden)
        hidden = Dropout(0.3)(hidden)
        hidden = Dense(64, activation="relu")(hidden)
        hidden = Dense(64, activation="relu")(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation("relu")(hidden)
        hidden = Dropout(0.3)(hidden)
        hidden = Dense(32, activation="relu")(hidden)
        hidden = Dense(32, activation="relu")(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation("relu")(hidden)
        return hidden

    def _build_model(self) -> Model:
        # Creating the two inputs
        source_input = Input(self._input_shape, name="source_input")
        destination_input = Input(self._input_shape, name="destination_input")

        # Build the source module
        source_module = self._sub_module(source_input)
        # Build the destination module
        destination_module = self._sub_module(destination_input)

        # Concatenating the two modules
        middle = Concatenate()([source_module, destination_module])

        # Creating the concatenation module
        hidden = Dropout(0.3)(middle)
        hidden = Dense(64, activation="relu")(hidden)
        hidden = Dense(64, activation="relu")(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation("relu")(hidden)
        hidden = Dropout(0.3)(hidden)
        hidden = Dense(32, activation="relu")(hidden)
        hidden = Dense(32, activation="relu")(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation("relu")(hidden)
        hidden = Dense(16, activation="relu")(hidden)
        hidden = Dense(8, activation="relu")(hidden)

        # Adding the model head
        head = Dense(1, activation="sigmoid")(hidden)

        # Building the multi-modal model.
        return Model(inputs=[source_input, destination_input], outputs=head)

    def fit_multi_modal(
        self,
        source_input_train: Union[List, np.ndarray],
        destination_input_train: Union[List, np.ndarray],
        output_train: Union[List, np.ndarray],
        source_input_test: Union[List, np.ndarray] = None,
        destination_input_test: Union[List, np.ndarray] = None,
        output_test: Union[List, np.ndarray] = None
    ) -> pd.DataFrame:
        # Converting input values to the format
        # to be used for a multi-modal model.
        train_x: Optional[Dict] = {
                "source_input": source_input_train,
                "destination_input": destination_input_train
            }
        train_y = output_train

        if all(d is not None for d in (source_input_test, destination_input_test, output_test)):
            test_x: Optional[Dict] = {
                "source_input": source_input_test,
                "destination_input": destination_input_test
            }
            test_y = output_test
        else:
            test_x = None
            test_y = None

        return self.fit(train_x, train_y, test_x, test_y)

    def predict_proba_multi_modal(
        self,
        source_input: Union[List, np.ndarray],
        destination_input: Union[List, np.ndarray]
    ):
        predictions = super().predict_proba(
            {
                "source_input": source_input,
                "destination_input": destination_input
            }
        )
        return np.hstack([
            1-predictions,
            predictions
        ])

    def predict_multi_modal(
        self,
        source_input: Union[List, np.ndarray],
        destination_input: Union[List, np.ndarray]
    ):
        return super().predict(
            {
                "source_input": source_input,
                "destination_input": destination_input
            }
        ).round().astype(int)