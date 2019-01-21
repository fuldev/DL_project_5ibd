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


class ReinforceClassicAgent(Agent):
    """
    Reinforce Classic Agent class for playing with it
    """

    def __init__(self, state_size: int, action_size: int, num_layers: int = 5, num_neuron_per_layer: int = 128) -> None:
        """
        Initializer for the `ReinforceClassicAgent` class
        :param state_size: ???
        :param action_size: ???
        :param num_layers: The number of layer for the Model
        :param num_neuron_per_layer: The number of neurons per layer for the model
        """
        self.brain = ReinforceClassicBrain(state_size, num_layers, num_neuron_per_layer, action_size)
        self.action_size = action_size
        self.episode_buffer = []

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `ReinforceClassicAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        vectorized_states = np.array([information_state.vectorize()] * len(available_actions))
        actions_vectorized = np.array([to_categorical(action, self.action_size) for action in available_actions])
        logits = self.brain.predict_policies(vectorized_states, actions_vectorized)
        sum = np.sum(logits)
        probabilities = np.reshape(logits / sum, (len(available_actions),))
        chosen_action = np.random.choice(available_actions, p=probabilities)
        transition = dict()
        transition['s'] = information_state.vectorize()
        transition['a'] = to_categorical(chosen_action, self.action_size)
        transition['r'] = 0.0
        transition['t'] = False
        self.episode_buffer.append(transition)
        return chosen_action

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `ReinforceClassicAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if not self.episode_buffer:
            return
        self.episode_buffer[len(self.episode_buffer) - 1]['r'] += reward
        self.episode_buffer[len(self.episode_buffer) - 1]['t'] |= terminal
        if terminal:
            states = np.array([transition['s'] for transition in self.episode_buffer])
            actions = np.array([transition['a'] for transition in self.episode_buffer])
            R = 0.0
            for t in reversed(range(len(self.episode_buffer))):
                R = self.episode_buffer[t]['r'] + 0.9 * R
                self.episode_buffer[t]['R'] = R
            advantages = np.array([transition['R'] for transition in self.episode_buffer])
            self.brain.train_policies(states, actions, advantages)
            self.episode_buffer = []


