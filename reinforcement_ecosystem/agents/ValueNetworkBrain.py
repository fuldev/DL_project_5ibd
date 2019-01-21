#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
from math import sqrt, log
from typing import Any, Iterable, Tuple

import numpy as np
from keras.activations import softmax
from keras.activations import relu, linear
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.layers.core import dense
import tensorflow as tf

from reinforcement_ecosystem.environments import InformationState, GameRunner, GameState, Agent



class ValueNetworkBrain:
    """
    ValueNetworkBrain class for ???
    """

    def __init__(self, state_size: int, num_players: int, num_layers: int = 2, num_neurons_per_layer: int = 512,
                 session: tf.Session = None) -> None:
        """
        Initializer for the `ValueNetworkBrain` class
        :param state_size: ???
        :param num_players: The number of players
        :param num_layers: The number of layer for the model
        :param num_neurons_per_layer: The number of neurons per layers
        :param session: The `tf.Session` to use
        """
        self.state_size = state_size
        self.num_players = num_players
        self.num_layers = num_layers
        self.num_neurons_per_layers = num_neurons_per_layer
        self.states_ph = tf.placeholder(shape=(None, state_size), dtype=tf.float64)
        self.target_values_ph = tf.placeholder(shape=(None, num_players), dtype=tf.float64)
        self.values_op, self.train_op = self.create_network()
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()
        if not self.session:
            self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_network(self) -> Any:
        """
        Build a neural network from the options passed as arguments
        :return: ???
        """
        hidden = self.states_ph
        for i in range(self.num_layers):
            hidden = dense(hidden, self.num_neurons_per_layers, activation=relu)
        values_op = dense(hidden, self.num_players, activation=linear)
        loss = tf.reduce_mean(tf.square(values_op - self.target_values_ph))
        train_op = tf.train.AdamOptimizer().minimize(loss)
        return values_op, train_op

    def predict_state(self, state: Any) -> Any:
        """
        ???
        :param state: ???
        :return: ???
        """
        return self.predict_states([state])[0]

    def predict_states(self, states: Any) -> Any:
        """
        ???
        :param states: ???
        :return: ???
        """
        return self.session.run(self.values_op, feed_dict={self.states_ph: states})

    def train(self, states: Any, target_values: Any) -> Any:
        """
        ???
        :param states: ???
        :param target_values: ???
        :return: ???
        """
        return self.session.run(self.train_op, feed_dict={
            self.states_ph: states,
            self.target_values_ph: target_values
        })