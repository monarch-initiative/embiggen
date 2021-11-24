"""Submodule providing an easy to use GNN model for edge prediction."""
from typing import Dict, List, Union, Optional, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # pylint: disable=import-error,no-name-in-module
from extra_keras_metrics import get_complete_binary_metrics
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Embedding, Flatten, GlobalAveragePooling1D, BatchNormalization  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.initializers import Initializer  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.regularizers import Regularizer  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.constraints import Constraint  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import Optimizer  # pylint: disable=import-error,no-name-in-module
import math

from embiggen.sequences import GNNEdgePredictionSequence, GNNBipartiteEdgePredictionSequence
from ensmallen import Graph
from embiggen.embedders.optimizers import apply_centralized_gradients
from embiggen.utils import validate_verbose, normalize_model_ragged_list_parameter, normalize_model_list_parameter
from tqdm.auto import tqdm
import tensorflow as tf


class EdgePredictionGraphNeuralNetwork:
    """Class proving a Graph Neural Network model for edge prediction."""

    def __init__(
        self,
        graph: Graph,
        node_features: Optional[Union[pd.DataFrame, List[pd.DataFrame], np.ndarray, List[np.ndarray]]] = None,
        node_feature_names: Union[str, List[str]] = "node_features",
        use_node_embedding: bool = True,
        node_embedding_size: int = 100,
        use_node_type_embedding: Union[bool, str] = "auto",
        node_type_embedding_size: int = 100,
        number_of_units_per_submodule_hidden_layer: Optional[Union[int, List[int], List[List[int]]]] = 128,
        number_of_units_per_body_hidden_layer: Optional[Union[int, List[int]]] = 64,
        number_of_units_per_header_hidden_layer: Optional[Union[int, List[int], List[List[int]]]] = 32,
        number_of_submodule_hidden_layers: Optional[Union[int, List[int]]] = 2,
        number_of_body_hidden_layers: Optional[int] = 2,
        number_of_header_hidden_layers: Optional[Union[int, List[int]]] = 2,
        activation_per_submodule_hidden_layer: Optional[Union[str, List[str]]] = "relu",
        activation_per_body_hidden_layer: Union[str, List[str]] = "relu",
        activation_per_header_hidden_layer: Optional[Union[str, List[str]]] = "relu",
        dropout_rate_per_submodule_hidden_layer: Optional[Union[float, List[float]]] = 0.5,
        dropout_rate_per_body_hidden_layer: Union[float, List[float]] = 0.4,
        dropout_rate_per_header_hidden_layer: Optional[Union[float, List[float]]] = 0.3,
        kernel_initializer_per_submodule_hidden_layer: Optional[Union[Union[str, Initializer], List[Union[str, Initializer]], List[List[Union[str, Initializer]]]]] = "glorot_uniform",
        kernel_initializer_per_body_hidden_layer: Union[Union[str, Initializer], List[Union[str, Initializer]]] = 'glorot_uniform',
        kernel_initializer_per_header_hidden_layer: Optional[Union[Union[str, Initializer], List[Union[str, Initializer]]]] = "glorot_uniform",
        bias_initializer_per_submodule_hidden_layer: Optional[Union[Union[str, Initializer], List[Union[str, Initializer]], List[List[Union[str, Initializer]]]]] = 'zeros',
        bias_initializer_per_body_hidden_layer: Union[Union[str, Initializer], List[Union[str, Initializer]]] = 'zeros',
        bias_initializer_per_header_hidden_layer: Optional[Union[Union[str, Initializer], List[Union[str, Initializer]]]] = 'zeros',
        kernel_regularizer_per_submodule_hidden_layer: Optional[Union[Regularizer, List[Regularizer], List[List[Regularizer]]]] = None,
        kernel_regularizer_per_body_hidden_layer: Union[Regularizer, List[Regularizer]] = None,
        kernel_regularizer_per_header_hidden_layer: Optional[Union[Regularizer, List[Regularizer]]] = None,
        bias_regularizer_per_submodule_hidden_layer: Optional[Union[Regularizer, List[Regularizer], List[List[Regularizer]]]] = None,
        bias_regularizer_per_body_hidden_layer: Union[Regularizer, List[Regularizer]] = None,
        bias_regularizer_per_header_hidden_layer: Optional[Union[Regularizer, List[Regularizer]]] = None,
        activity_regularizer_per_submodule_hidden_layer: Optional[Union[Regularizer, List[Regularizer], List[List[Regularizer]]]] = None,
        activity_regularizer_per_body_hidden_layer: Union[Regularizer, List[Regularizer]] = None,
        activity_regularizer_per_header_hidden_layer: Optional[Union[Regularizer, List[Regularizer]]] = None,
        kernel_constraint_per_submodule_hidden_layer: Optional[Union[Union[str, Constraint], List[Union[str, Constraint]], List[List[Union[str, Constraint]]]]] = None,
        kernel_constraint_per_body_hidden_layer: Union[Union[str, Constraint], List[Union[str, Constraint]]] = None,
        kernel_constraint_per_header_hidden_layer: Optional[Union[Union[str, Constraint], List[Union[str, Constraint]]]] = None,
        bias_constraint_per_submodule_hidden_layer: Optional[Union[Union[str, Constraint], List[Union[str, Constraint]], List[List[Union[str, Constraint]]]]] = None,
        bias_constraint_per_body_hidden_layer: Union[Union[str, Constraint], List[Union[str, Constraint]]] = None,
        bias_constraint_per_header_hidden_layer: Optional[Union[Union[str, Constraint], List[Union[str, Constraint]]]] = None,
        use_class_weights: bool = False,
        use_batch_normalization: bool = True,
        use_gradient_centralization: bool = True,
        optimizer: Union[str, Optimizer] = "nadam",
        model_name: Optional[str] = None
    ):
        """Create new Abstract Graph Convolutional Network.

        Parameters
        -------------------------------
        TODO!

        Raises
        ---------------------------------
        TODO!
        """
        self._nodes_number = graph.get_nodes_number()
        if graph.has_node_types():
            self._node_types_number = graph.get_node_types_number()
        else:
            self._node_types_number = None
        self._use_node_embedding = use_node_embedding
        if use_node_type_embedding == "auto":
            use_node_type_embedding = graph.has_node_types(
            ) and not graph.has_unknown_node_types()
        self._use_node_type_embedding = use_node_type_embedding

        self._node_embedding_size = node_embedding_size
        self._node_type_embedding_size = node_type_embedding_size

        # First we need to normalize the node features to a list
        if node_features is not None and isinstance(node_features, (np.ndarray, pd.DataFrame)):
            node_features = [
                node_features
            ]
        if node_features is None:
            node_features = []

        self._node_features_number = [
            nfs.shape[1:]
            for nfs in node_features
        ]

        if isinstance(node_feature_names, str):
            node_feature_names = [
                node_feature_names
            ]

        if len(node_feature_names) != len(node_features):
            raise ValueError(
                (
                    "The provided node feature names list length is `{}` "
                    "but the number of provided node features dataframes is `{}`."
                ).format(
                    len(node_feature_names),
                    len(node_features)
                )
            )

        if isinstance(node_feature_names, str):
            node_feature_names = [
                node_feature_names
            ]

        self._node_feature_names = node_feature_names

        submodules_number = len(
            node_features) + int(self._use_node_embedding) + int(self._use_node_type_embedding)

        self._node_features = node_features

        ####################################################
        # Normalize the provided parameters for submodules #
        ####################################################

        self._number_of_submodule_hidden_layers = normalize_model_list_parameter(
            number_of_submodule_hidden_layers,
            submodules_number,
            int
        )
        self._number_of_units_per_submodule_hidden_layer = normalize_model_ragged_list_parameter(
            number_of_units_per_submodule_hidden_layer,
            submodules_number,
            self._number_of_submodule_hidden_layers,
            int,
            number_of_units_per_body_hidden_layer
        )
        self._activation_per_submodule_hidden_layer = normalize_model_ragged_list_parameter(
            activation_per_submodule_hidden_layer,
            submodules_number,
            self._number_of_submodule_hidden_layers,
            (str, type(None)),
            activation_per_body_hidden_layer
        )
        self._dropout_rate_per_submodule_hidden_layer = normalize_model_ragged_list_parameter(
            dropout_rate_per_submodule_hidden_layer,
            submodules_number,
            self._number_of_submodule_hidden_layers,
            (float, type(None)),
            dropout_rate_per_body_hidden_layer
        )
        self._kernel_initializer_per_submodule_hidden_layer = normalize_model_ragged_list_parameter(
            kernel_initializer_per_submodule_hidden_layer,
            submodules_number,
            self._number_of_submodule_hidden_layers,
            (Initializer, str, type(None)),
            kernel_initializer_per_body_hidden_layer
        )
        self._bias_initializer_per_submodule_hidden_layer = normalize_model_ragged_list_parameter(
            bias_initializer_per_submodule_hidden_layer,
            submodules_number,
            self._number_of_submodule_hidden_layers,
            (Initializer, str, type(None)),
            bias_initializer_per_body_hidden_layer
        )
        self._kernel_regularizer_per_submodule_hidden_layer = normalize_model_ragged_list_parameter(
            kernel_regularizer_per_submodule_hidden_layer,
            submodules_number,
            self._number_of_submodule_hidden_layers,
            (Regularizer, type(None)),
            kernel_regularizer_per_submodule_hidden_layer
        )
        self._bias_regularizer_per_submodule_hidden_layer = normalize_model_ragged_list_parameter(
            bias_regularizer_per_submodule_hidden_layer,
            submodules_number,
            self._number_of_submodule_hidden_layers,
            (Regularizer, type(None)),
            bias_regularizer_per_body_hidden_layer
        )
        self._activity_regularizer_per_submodule_hidden_layer = normalize_model_ragged_list_parameter(
            activity_regularizer_per_submodule_hidden_layer,
            submodules_number,
            self._number_of_submodule_hidden_layers,
            (Regularizer, type(None)),
            activity_regularizer_per_body_hidden_layer
        )
        self._kernel_constraint_per_submodule_hidden_layer = normalize_model_ragged_list_parameter(
            kernel_constraint_per_submodule_hidden_layer,
            submodules_number,
            self._number_of_submodule_hidden_layers,
            (Constraint, str, type(None)),
            kernel_constraint_per_body_hidden_layer
        )
        self._bias_constraint_per_submodule_hidden_layer = normalize_model_ragged_list_parameter(
            bias_constraint_per_submodule_hidden_layer,
            submodules_number,
            self._number_of_submodule_hidden_layers,
            (Constraint, str, type(None)),
            bias_constraint_per_body_hidden_layer
        )

        ##############################################
        # Normalize the provided parameters for body #
        ##############################################

        self._number_of_body_hidden_layers = number_of_body_hidden_layers

        self._number_of_units_per_body_hidden_layer = normalize_model_list_parameter(
            number_of_units_per_body_hidden_layer,
            number_of_body_hidden_layers,
            int
        )

        self._activation_per_body_hidden_layer = normalize_model_list_parameter(
            activation_per_body_hidden_layer,
            number_of_body_hidden_layers,
            (str, type(None))
        )
        self._dropout_rate_per_body_hidden_layer = normalize_model_list_parameter(
            dropout_rate_per_body_hidden_layer,
            number_of_body_hidden_layers,
            (float, type(None))
        )
        self._kernel_initializer_per_body_hidden_layer = normalize_model_list_parameter(
            kernel_initializer_per_body_hidden_layer,
            number_of_body_hidden_layers,
            (Initializer, str, type(None))
        )
        self._bias_initializer_per_body_hidden_layer = normalize_model_list_parameter(
            bias_initializer_per_body_hidden_layer,
            number_of_body_hidden_layers,
            (Initializer, str, type(None))
        )
        self._kernel_regularizer_per_body_hidden_layer = normalize_model_list_parameter(
            kernel_regularizer_per_body_hidden_layer,
            number_of_body_hidden_layers,
            (Regularizer, type(None))
        )
        self._bias_regularizer_per_body_hidden_layer = normalize_model_list_parameter(
            bias_regularizer_per_body_hidden_layer,
            number_of_body_hidden_layers,
            (Regularizer, type(None))
        )
        self._activity_regularizer_per_body_hidden_layer = normalize_model_list_parameter(
            activity_regularizer_per_body_hidden_layer,
            number_of_body_hidden_layers,
            (Regularizer, type(None))
        )
        self._kernel_constraint_per_body_hidden_layer = normalize_model_list_parameter(
            kernel_constraint_per_body_hidden_layer,
            number_of_body_hidden_layers,
            (Constraint, str, type(None))
        )
        self._bias_constraint_per_body_hidden_layer = normalize_model_list_parameter(
            bias_constraint_per_body_hidden_layer,
            number_of_body_hidden_layers,
            (Constraint, str, type(None))
        )

        ##############################################
        # Normalize the provided parameters for header #
        ##############################################

        self._number_of_header_hidden_layers = number_of_header_hidden_layers

        self._number_of_units_per_header_hidden_layer = normalize_model_list_parameter(
            number_of_units_per_header_hidden_layer,
            number_of_header_hidden_layers,
            int
        )

        self._activation_per_header_hidden_layer = normalize_model_list_parameter(
            activation_per_header_hidden_layer,
            number_of_header_hidden_layers,
            (str, type(None))
        )
        self._dropout_rate_per_header_hidden_layer = normalize_model_list_parameter(
            dropout_rate_per_header_hidden_layer,
            number_of_header_hidden_layers,
            (float, type(None))
        )
        self._kernel_initializer_per_header_hidden_layer = normalize_model_list_parameter(
            kernel_initializer_per_header_hidden_layer,
            number_of_header_hidden_layers,
            (Initializer, str, type(None))
        )
        self._bias_initializer_per_header_hidden_layer = normalize_model_list_parameter(
            bias_initializer_per_header_hidden_layer,
            number_of_header_hidden_layers,
            (Initializer, str, type(None))
        )
        self._kernel_regularizer_per_header_hidden_layer = normalize_model_list_parameter(
            kernel_regularizer_per_header_hidden_layer,
            number_of_header_hidden_layers,
            (Regularizer, type(None))
        )
        self._bias_regularizer_per_header_hidden_layer = normalize_model_list_parameter(
            bias_regularizer_per_header_hidden_layer,
            number_of_header_hidden_layers,
            (Regularizer, type(None))
        )
        self._activity_regularizer_per_header_hidden_layer = normalize_model_list_parameter(
            activity_regularizer_per_header_hidden_layer,
            number_of_header_hidden_layers,
            (Regularizer, type(None))
        )
        self._kernel_constraint_per_header_hidden_layer = normalize_model_list_parameter(
            kernel_constraint_per_header_hidden_layer,
            number_of_header_hidden_layers,
            (Constraint, str, type(None))
        )
        self._bias_constraint_per_header_hidden_layer = normalize_model_list_parameter(
            bias_constraint_per_header_hidden_layer,
            number_of_header_hidden_layers,
            (Constraint, str, type(None))
        )

        ############################################
        # Handle the optimizer-releated parameters #
        ############################################

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if use_gradient_centralization:
            apply_centralized_gradients(optimizer)

        self._optimizer = optimizer

        ######################################################
        # Handle the other parameters relative to the model. #
        ######################################################

        self._use_class_weights = use_class_weights
        self._use_batch_normalization = use_batch_normalization

        #########################################
        # Create name of the model if necessary #
        #########################################

        if model_name is None:
            model_name = "{}EdgePredictionGNN".format(graph.get_name())
        self._model_name = model_name

        ###################
        # Build the model #
        ###################

        self._model = self._build_model()

        #####################
        # Compile the model #
        #####################

        self._compile_model()

    def _build_submodule(self) -> Model:
        """Create the submodule for the source or destination node."""

        #################################################
        # Create the input layers for the node features #
        #################################################

        input_features = [
            Input(
                shape=features_size,
                name=node_feature_name
            )
            for node_feature_name, features_size in zip(
                self._node_feature_names,
                self._node_features_number
            )
        ]
        # We copy the single objects
        # because we want ton create a second list
        # since the two lists will diverge in the following sections.
        input_layers = [
            input_feature
            for input_feature in input_features
        ]

        ##############################################################
        # Then we need to create the input layers for the node ids   #
        # if a node embedding model is enabled in this model.        #
        ##############################################################

        if self._use_node_embedding:
            node_ids = Input(
                shape=(1, ),
                name="NodeID"
            )
            nodes_embedding = Embedding(
                self._nodes_number, self._node_embedding_size,
                input_length=1,
                name="NodeEmbedding"
            )
            flatten = Flatten()
            node_embedding = flatten(nodes_embedding(node_ids))
            input_features.append(node_embedding)
            input_layers.append(node_ids)

        ##############################################################
        # Then we need to create the input layers for the node types #
        # if they are requested and a node type embedding layer      #
        # is enabled in this model                                   #
        ##############################################################

        if self._use_node_type_embedding:
            node_type_ids = Input(
                shape=(None,),
                name="NodeTypes"
            )
            node_types_embedding = Embedding(
                self._node_types_number + 1, self._node_type_embedding_size,
                mask_zero=True,
                input_length=None,
                name="NodeTypeEmbedding"
            )
            global_average = GlobalAveragePooling1D()
            node_type_embedding = global_average(
                node_types_embedding(node_type_ids)
            )
            input_features.append(node_type_embedding)
            input_layers.append(node_type_ids)

        ##########################################
        # If needed, create the input submodules #
        ##########################################

        if len(input_features) > 1:
            input_submodules = []
            for (
                nfs,
                number_of_units_per_hidden_layer,
                activation_per_hidden_layer,
                dropout_rate_per_hidden_layer,
                kernel_initializer_per_hidden_layer,
                bias_initializer_per_hidden_layer,
                kernel_regularizer_per_hidden_layer,
                bias_regularizer_per_hidden_layer,
                activity_regularizer_per_hidden_layer,
                kernel_constraint_per_hidden_layer,
                bias_constraint_per_hidden_layer
            ) in zip(
                input_features,
                self._number_of_units_per_submodule_hidden_layer,
                self._activation_per_submodule_hidden_layer,
                self._dropout_rate_per_submodule_hidden_layer,
                self._kernel_initializer_per_submodule_hidden_layer,
                self._bias_initializer_per_submodule_hidden_layer,
                self._kernel_regularizer_per_submodule_hidden_layer,
                self._bias_regularizer_per_submodule_hidden_layer,
                self._activity_regularizer_per_submodule_hidden_layer,
                self._kernel_constraint_per_submodule_hidden_layer,
                self._bias_constraint_per_submodule_hidden_layer
            ):
                hidden = nfs
                for (
                    number_of_units,
                    activation,
                    dropout_rate,
                    kernel_initializer,
                    bias_initializer,
                    kernel_regularizer,
                    bias_regularizer,
                    activity_regularizer,
                    kernel_constraint,
                    bias_constraint
                ) in zip(
                    number_of_units_per_hidden_layer,
                    activation_per_hidden_layer,
                    dropout_rate_per_hidden_layer,
                    kernel_initializer_per_hidden_layer,
                    bias_initializer_per_hidden_layer,
                    kernel_regularizer_per_hidden_layer,
                    bias_regularizer_per_hidden_layer,
                    activity_regularizer_per_hidden_layer,
                    kernel_constraint_per_hidden_layer,
                    bias_constraint_per_hidden_layer
                ):
                    hidden = Dense(
                        units=number_of_units,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                    )(hidden)
                    if self._use_batch_normalization:
                        hidden = BatchNormalization()(hidden)
                    if dropout_rate > 0:
                        hidden = Dropout(rate=dropout_rate)(hidden)

                input_submodules.append(hidden)

            # Now that we have created all the submodules, we need to concatenate
            # the various submodules.
            node_features = Concatenate()(input_submodules)
        else:
            node_features = input_features[0]

        ######################################
        # Create the body segment of the GNN #
        ######################################

        hidden = node_features

        for (
            number_of_units,
            activation,
            dropout_rate,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint
        ) in zip(
            self._number_of_units_per_body_hidden_layer,
            self._activation_per_body_hidden_layer,
            self._dropout_rate_per_body_hidden_layer,
            self._kernel_initializer_per_body_hidden_layer,
            self._bias_initializer_per_body_hidden_layer,
            self._kernel_regularizer_per_body_hidden_layer,
            self._bias_regularizer_per_body_hidden_layer,
            self._activity_regularizer_per_body_hidden_layer,
            self._kernel_constraint_per_body_hidden_layer,
            self._bias_constraint_per_body_hidden_layer
        ):
            hidden = Dense(
                units=number_of_units,
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
            )(hidden)
            if self._use_batch_normalization:
                hidden = BatchNormalization()(hidden)
            if dropout_rate > 0:
                hidden = Dropout(rate=dropout_rate)(hidden)

        ###########################
        # Finally build the model #
        ###########################

        return Model(
            inputs=input_layers,
            outputs=hidden,
            name="{}Submodule".format(self._model_name)
        )

    def _build_model(self):
        """Create new GNN model."""

        #######################################
        # Create the submodel for the Siamese #
        #######################################
        submodel = self._build_submodule()

        #################################################
        # Create the input layers for the node features #
        #################################################

        # Specifically, we expect to have two set of input features for the
        # source and destination nodes.
        subject_input_layers = []
        object_input_layers = []

        for side_of_edge, input_layer_list in (
            ("Subject", subject_input_layers),
            ("Object", object_input_layers)
        ):
            for node_feature_name, features_size in zip(
                self._node_feature_names,
                self._node_features_number
            ):
                node_features_input = Input(
                    shape=features_size,
                    name="{}_{}".format(node_feature_name, side_of_edge)
                )
                input_layer_list.append(node_features_input)

        ##############################################################
        # Then we need to create the input layers for the node ids   #
        # if a node embedding model is enabled in this model.        #
        ##############################################################

        if self._use_node_embedding:
            subject_input_layers.append(Input(
                shape=(1, ),
                name="Subjects"
            ))
            object_input_layers.append(Input(
                shape=(1, ),
                name="Objects"
            ))

        ##############################################################
        # Then we need to create the input layers for the node types #
        # if they are requested and a node type embedding layer      #
        # is enabled in this model                                   #
        ##############################################################

        if self._use_node_type_embedding:
            subject_input_layers.append(Input(
                shape=(None,),
                name="SubjectNodeTypes"
            ))
            object_input_layers.append(Input(
                shape=(None,),
                name="ObjectNodeTypes"
            ))

        ############################################
        # Create the complete list of input layers #
        ############################################
        input_layers = subject_input_layers + object_input_layers

        ##############################################
        # Get the features from the Siamese submodel #
        ##############################################

        subject_submodel = submodel(subject_input_layers)
        object_submodel = submodel(object_input_layers)

        concatenation = Concatenate()([
            subject_submodel,
            object_submodel
        ])

        ###########################
        # Create the header model #
        ###########################

        hidden = concatenation
        for (
            number_of_units,
            activation,
            dropout_rate,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint
        ) in zip(
            self._number_of_units_per_header_hidden_layer,
            self._activation_per_header_hidden_layer,
            self._dropout_rate_per_header_hidden_layer,
            self._kernel_initializer_per_header_hidden_layer,
            self._bias_initializer_per_header_hidden_layer,
            self._kernel_regularizer_per_header_hidden_layer,
            self._bias_regularizer_per_header_hidden_layer,
            self._activity_regularizer_per_header_hidden_layer,
            self._kernel_constraint_per_header_hidden_layer,
            self._bias_constraint_per_header_hidden_layer
        ):
            hidden = Dense(
                units=number_of_units,
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
            )(hidden)
            if self._use_batch_normalization:
                hidden = BatchNormalization()(hidden)
            if dropout_rate > 0:
                hidden = Dropout(rate=dropout_rate)(hidden)

        hidden = Dense(1, activation="sigmoid")(hidden)

        ###########################
        # Finally build the model #
        ###########################

        return Model(
            inputs=input_layers,
            outputs=hidden,
            name=self._model_name
        )

    def _compile_model(self):
        """Compile model."""
        self._model.compile(
            loss='binary_crossentropy',
            optimizer=self._optimizer,
            weighted_metrics=get_complete_binary_metrics()
        )

    @property
    def name(self) -> str:
        return self._model.name

    def summary(self, *args, **kwargs):
        """Print model summary."""
        self._model.summary(*args, **kwargs)

    def plot(
        self,
        show_shapes: bool = True,
        **kwargs: Dict
    ):
        """Plot model structure as Dot plot.

        Parameters
        -------------------------
        show_shapes: bool = True
            Whether to show the input and output shapes of the layers.
        **kwargs: Dict
            Parameters to be passed to the plot model call.
        """
        return tf.keras.utils.plot_model(
            self._model,
            show_shapes=show_shapes,
            **kwargs
        )

    def fit(
        self,
        training_graph: Graph,
        batch_size: int = 2**15,
        negative_samples_rate: float = 0.8,
        early_stopping_min_delta: float = 0.00001,
        early_stopping_patience: int = 10,
        reduce_lr_min_delta: float = 0.00001,
        reduce_lr_patience: int = 5,
        validation_graph: Optional[Graph] = None,
        validation_batch_size: int = 2**15,
        epochs: int = 1000,
        early_stopping_monitor: str = "val_loss",
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "val_loss",
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.1,
        verbose: int = 2,
        **kwargs: Dict
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history."""
        try:
            from tqdm.keras import TqdmCallback
            traditional_verbose = False
        except AttributeError:
            traditional_verbose = True
        verbose = validate_verbose(verbose)

        training_sequence = GNNEdgePredictionSequence(
            training_graph,
            self._node_features,
            use_node_types=self._use_node_type_embedding,
            use_edge_metrics=False,
            return_node_ids=self._use_node_embedding,
            batch_size=batch_size,
            negative_samples_rate=negative_samples_rate
        )
        training_steps = training_sequence.steps_per_epoch
        training_sequence = training_sequence.into_dataset().repeat()

        if validation_graph is not None:
            validation_data = GNNEdgePredictionSequence(
                validation_graph,
                self._node_features,
                use_node_types=self._use_node_type_embedding,
                use_edge_metrics=False,
                return_node_ids=self._use_node_embedding,
                batch_size=validation_batch_size,
                negative_samples_rate=negative_samples_rate,
                graph_to_avoid=training_graph
            )
            validation_steps = validation_data.steps_per_epoch
            validation_data = validation_data.into_dataset().repeat()
        else:
            validation_steps = None
            validation_data = None

        callbacks = kwargs.pop("callbacks", ())
        return pd.DataFrame(self._model.fit(
            training_sequence,
            epochs=epochs,
            steps_per_epoch=training_steps,
            verbose=traditional_verbose and verbose > 0,
            validation_data=validation_data,
            validation_steps=validation_steps,
            class_weight= {
                False: training_graph.get_density(),
                True: 1.0 - training_graph.get_density()
            } if self._use_class_weights else None,
            shuffle=False,
            callbacks=[
                EarlyStopping(
                    monitor=early_stopping_monitor,
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    mode=early_stopping_mode,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor=reduce_lr_monitor,
                    min_delta=reduce_lr_min_delta,
                    patience=reduce_lr_patience,
                    factor=reduce_lr_factor,
                    mode=reduce_lr_mode,
                ),
                *((TqdmCallback(
                    verbose=verbose-1,
                ),)
                    if not traditional_verbose and verbose > 0 else ()),
                *callbacks
            ],
            **kwargs
        ).history)

    def predict_from_node_ids(
        self,
        graph: Graph,
        source_node_ids: np.ndarray,
        destination_node_ids: np.ndarray,
        minimum_score: float = 0.9,
        always_return_existing_edges: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Run predictions on the described bipartite graph.

        Parameters
        ---------------------------
        graph: Graph
            The graph from where to sample the nodes.
        source_node_ids: np.ndarray
            Vector of source node IDs in bipartite graph.
        destination_node_ids: np.ndarray
            Vector of destination node IDs in bipartite graph.
        minimum_score: float = 0.95
            Since the edges to return are generally a very high
            number, we usually want to filter.
        always_return_existing_edges: bool = True
            Whether to always return scores relative to existing edges.
        verbose: bool = True
            Whether to show the loading bars.
        """
        sequence = GNNBipartiteEdgePredictionSequence(
            graph,
            sources=source_node_ids,
            destinations=destination_node_ids,
            node_features=self._node_features,
            use_node_types=self._use_node_type_embedding,
            return_node_ids=self._use_node_embedding,
            return_labels=False
        )

        predictions = self._model.predict(
            sequence,
            verbose=verbose
        ).flatten()

        source_node_names = [
            graph.get_node_name_from_node_id(node_id)
            for node_id in tqdm(
                source_node_ids,
                desc="Retrieving source node names",
                dynamic_ncols=True,
                leave=False,
                disable=not verbose
            )
        ]
        destination_node_names = [
            graph.get_node_name_from_node_id(node_id)
            for node_id in tqdm(
                destination_node_ids,
                desc="Retrieving destination node names",
                dynamic_ncols=True,
                leave=False,
                disable=not verbose
            )
        ]

        tiled_source_node_names = [
            source_node_name
            for source_node_name in source_node_names
            for _ in range(len(destination_node_names))
        ]

        tiled_destination_node_names = [
            destination_node_name
            for _ in range(len(source_node_names))
            for destination_node_name in destination_node_names
        ]

        exists = [
            graph.has_edge_from_node_names(
                source_node_name,
                destination_node_name
            )
            for source_node_name, destination_node_name in tqdm(
                zip(
                    tiled_source_node_names,
                    tiled_destination_node_names
                ),
                total=len(tiled_source_node_names),
                desc="Computing whether edge exists",
                leave=False,
                dynamic_ncols=True
            )
        ]

        if minimum_score > 0:
            (
                tiled_source_node_names,
                tiled_destination_node_names,
                predictions,
                exists
            ) = list(zip(*(
                (src, dst, pred, exist)
                for (src, dst, pred, exist) in tqdm(
                    zip(
                        tiled_source_node_names,
                        tiled_destination_node_names,
                        predictions,
                        exists
                    ),
                    total=len(tiled_source_node_names),
                    desc="Filtering edges",
                    leave=False,
                    dynamic_ncols=True
                )
                if pred >= minimum_score or exist and always_return_existing_edges
            )))

        return pd.DataFrame({
            "source_node_name": tiled_source_node_names,
            "destination_node_name": tiled_destination_node_names,
            "predictions": predictions,
            "exists": exists
        })

    def predict_from_node_types(
        self,
        graph: Graph,
        source_node_type_name: str,
        destination_node_type_name: str,
        minimum_score: float = 0.9,
        always_return_existing_edges: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Run predictions on the described bipartite graph.

        Parameters
        ---------------------------
        graph: Graph
            The graph from where to sample the nodes.
        source_node_type_name: str
            The node type describing the source nodes.
        destination_node_type_name: str
            The node type describing the destination nodes.
        minimum_score: float = 0.95
            Since the edges to return are generally a very high
            number, we usually want to filter.
        always_return_existing_edges: bool = True
            Whether to always return scores relative to existing edges.
        verbose: bool = True
            Whether to show the loading bars.
        """
        return self.predict_from_node_ids(
            graph,
            source_node_ids=graph.get_node_ids_from_node_type_name(
                source_node_type_name
            ),
            destination_node_ids=graph.get_node_ids_from_node_type_name(
                destination_node_type_name
            ),
            minimum_score=minimum_score,
            always_return_existing_edges=always_return_existing_edges,
            verbose=verbose
        )

    def evaluate_from_node_ids(
        self,
        graph: Graph,
        source_node_ids: np.ndarray,
        destination_node_ids: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Run evaluations on the described bipartite graph.

        Parameters
        ---------------------------
        graph: Graph
            The graph from where to sample the nodes.
        source_node_ids: np.ndarray
            Vector of source node IDs in bipartite graph.
        destination_node_ids: np.ndarray
            Vector of destination node IDs in bipartite graph.
        verbose: bool = True
            Whether to show the loading bars.
        """
        sequence = GNNBipartiteEdgePredictionSequence(
            graph,
            sources=source_node_ids,
            destinations=destination_node_ids,
            node_features=self._node_features,
            use_node_types=self._use_node_type_embedding,
            return_node_ids=self._use_node_embedding,
            return_labels=True
        )

        return dict(zip(
            self._model.metrics_names,
            self._model.evaluate(
                sequence,
                verbose=verbose
            )
        ))

    def evaluate_from_node_types(
        self,
        graph: Graph,
        source_node_type_name: str,
        destination_node_type_name: str,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Run predictions on the described bipartite graph.

        Parameters
        ---------------------------
        graph: Graph
            The graph from where to sample the nodes.
        source_node_type_name: str
            The node type describing the source nodes.
        destination_node_type_name: str
            The node type describing the destination nodes.
        verbose: bool = True
            Whether to show the loading bars.
        """
        return self.evaluate_from_node_ids(
            graph,
            source_node_ids=graph.get_node_ids_from_node_type_name(
                source_node_type_name
            ),
            destination_node_ids=graph.get_node_ids_from_node_type_name(
                destination_node_type_name
            ),
            verbose=verbose
        )
