#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Any, Iterable

import numpy as np
from keras import Input, Model
from keras.activations import tanh, sigmoid
from keras.layers import Dense, concatenate
from keras.optimizers import adam
from keras.utils import to_categorical
import keras.backend as K

from reinforcement_ecosystem.environments import InformationState, Agent


class ReinforceClassicBrain:

    def __init__(self, state_size: int, num_layers: int, num_neuron_per_layer: int, action_size: int):
        """
        Initializer for `ReinforceClassicBrain`
        :param state_size: ???
        :param num_layers: The number of layer of the model
        :param num_neuron_per_layer: The number of neuron per layer
        :param action_size: The maximum number of action
        """
        self.state_size = state_size
        self.num_layers = num_layers
        self.num_neuron_per_layer = num_neuron_per_layer
        self.action_size = action_size
        self.model = self.create_model()

    def create_model(self) -> Model:
        """
        Create a Model from the internal state of the class
        :return: A Keras `Model`
        """
        input_state = Input(shape=(self.state_size,))
        input_action = Input(shape=(self.action_size,))
        hidden = concatenate([input_state, input_action])
        for i in range(self.num_layers):
            hidden = Dense(self.num_neuron_per_layer, activation=tanh)(hidden)
        policy = Dense(1, activation=sigmoid)(hidden)
        model = Model([input_state, input_action], policy)
        model.compile(loss=ReinforceClassicBrain.reinforce_loss, optimizer=adam())
        return model

    def predict_policy(self, state: Any, action: Any) -> Any:
        """
        ???
        :param state: ???
        :param action: ???
        :return: ???
        """
        return self.model.predict([np.array([state]), np.array([action])])[0]

    def predict_policies(self, states: Any, actions: Any) -> Any:
        """
        ???
        :param states: ???
        :param actions: ???
        :return: ???
        """
        return self.model.predict([states, actions])

    def reinforce_loss(self, y_true: Any, y_pred: Any) -> Any:
        """
        ???
        :param y_true: ???
        :param y_pred: ???
        :return: ???
        """
        return K.mean(-K.log(y_pred) * y_true)

    def train_policy(self, state: Any, action: Any, advantage: Any) -> None:
        """
        ???
        :param state: ???
        :param action: ???
        :param advantage: ???
        :return: ???
        """
        self.model.train_on_batch([np.array([state]), np.array([action])], np.array([advantage]))

    def train_policies(self, states: Any, actions: Any, advantages: Any) -> None:
        """
        ???
        :param states: ???
        :param actions: ???
        :param advantages: ???
        """
        self.model.train_on_batch([states, actions], advantages)
