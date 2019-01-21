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

class ReinforceClassicWithMultipleTrajectoriesAgent(Agent):
    """
    Reinforce Classic With Multiple Trajectories  Agent class for playing with it
    """

    def __init__(self, state_size: int, action_size: int, num_layers: int = 5, num_neuron_per_layer: int = 128,
                 train_every_X_trajectories: int = 16):
        """
        Initializer for the `ReinforceClassicWithMultipleTrajectoriesAgent` class
        :param state_size: ???
        :param action_size: ???
        :param num_layers: The number of layer for the Model
        :param num_neuron_per_layer: The number of neuron per layer for the model
        :param train_every_X_trajectories: ???
        """
        self.brain = ReinforceClassicBrain(state_size, num_layers, num_neuron_per_layer, action_size)
        self.action_size = action_size
        self.trajectories = []
        self.current_trajectory_buffer = []
        self.train_every_X_trajectory = train_every_X_trajectories

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `ReinforceClassicWithMultipleTrajectoriesAgent`
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
        self.current_trajectory_buffer.append(transition)
        return chosen_action

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `ReinforceClassicWithMultipleTrajectoriesAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if not self.current_trajectory_buffer:
            return
        self.current_trajectory_buffer[len(self.current_trajectory_buffer) - 1]['r'] += reward
        self.current_trajectory_buffer[len(self.current_trajectory_buffer) - 1]['t'] |= terminal
        if terminal:
            R = 0.0
            for t in reversed(range(len(self.current_trajectory_buffer))):
                R = self.current_trajectory_buffer[t]['r'] + 0.9 * R
                self.current_trajectory_buffer[t]['R'] = R
            self.trajectories.append(self.current_trajectory_buffer)
            self.current_trajectory_buffer = []
            if len(self.trajectories) == self.train_every_X_trajectory:
                states = np.array([transition['s'] for trajectory in self.trajectories for transition in trajectory])
                actions = np.array([transition['a'] for trajectory in self.trajectories for transition in trajectory])
                advantages = np.array([transition['R'] for trajectory in self.trajectories for transition in trajectory])
                self.brain.train_policies(states, actions, advantages)
                self.trajectories = []
