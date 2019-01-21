#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Any, Iterable

import numpy as np
import tensorflow as tf
from keras.constraints import max_norm

from reinforcement_ecosystem.environments import InformationState, Agent

class PPOWithMultipleTrajectoriesMultiOutputsAgent(Agent):
    """
    PPOWithMultipleTrajectoriesMultiOutputsAgent class for playing with it
    """

    def __init__(self, state_size: int, action_size: int, num_layers: int = 5, num_neuron_per_layer: int = 64,
                 train_every_x_trajectories: int = 64, gamma: float = 0.9999,
                 num_epochs: int = 4, batch_size: int = 256):
        """
        Initializer for PPOWithMultipleTrajectoriesMultiOutputsAgent clas
        :param state_size: ???
        :param action_size: ???
        :param num_layers: The number of layer for the model
        :param num_neuron_per_layer: The number of neuron per layer for the model
        :param train_every_x_trajectories: ???
        :param gamma: ???
        :param num_epochs: The number of epochs to train the model on
        :param batch_size: The batch size for the the model to work with
        """
        self.brain = PPOBrain(state_size, num_layers, num_neuron_per_layer, action_size)
        self.action_size = action_size
        self.gamma = gamma
        self.trajectories = []
        self.current_trajectory_buffer = []
        self.train_every_x_trajectory = train_every_x_trajectories
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `PPOWithMultipleTrajectoriesMultiOutputsAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        available_actions = list(available_actions)
        num_actions = len(available_actions)
        vectorized_state = information_state.vectorize()
        full_actions_probability, value = self.brain.predict_policy_and_value(vectorized_state)
        available_actions_probabilities = full_actions_probability[available_actions]
        sum_available_action_probabilities = np.sum(available_actions_probabilities)
        if sum_available_action_probabilities > 0.0000001:  # just in case all is zero, but unlikely
            probabilities = available_actions_probabilities / sum_available_action_probabilities
            chosen_index = np.random.choice(list(range(num_actions)), p=probabilities)
            chosen_action = available_actions[chosen_index]
        else:
            print("No action eligible, this should be extremely rare")
            chosen_index = np.random.choice(list(range(num_actions)))
            chosen_action = available_actions[chosen_index]
        transition = dict()
        transition['s'] = vectorized_state
        transition['a'] = chosen_action
        transition['r'] = 0.0
        transition['t'] = False
        transition['p_old'] = full_actions_probability.tolist()
        self.current_trajectory_buffer.append(transition)
        return chosen_action

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `PPOWithMultipleTrajectoriesMultiOutputsAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if not self.current_trajectory_buffer:
            return
        self.current_trajectory_buffer[len(self.current_trajectory_buffer) - 1]['r'] += reward
        self.current_trajectory_buffer[len(self.current_trajectory_buffer) - 1]['t'] |= terminal
        if terminal:
            accumulated_reward = 0.0
            for t in reversed(range(len(self.current_trajectory_buffer))):
                accumulated_reward = self.current_trajectory_buffer[t]['r'] + self.gamma * accumulated_reward
                self.current_trajectory_buffer[t]['R'] = accumulated_reward
            self.trajectories.append(self.current_trajectory_buffer)
            self.current_trajectory_buffer = []
            if len(self.trajectories) == self.train_every_x_trajectory:
                transitions = [transition for trajectory in self.trajectories for transition in trajectory]
                states = np.array(
                    [transition['s'] for transition in transitions])
                accumulated_rewards = np.array(
                    [transition['R'] for transition in transitions])
                old_policies = np.array(
                    [transition['p_old'] for transition in transitions])
                num_samples = states.shape[0]
                batch_size = min(self.batch_size, num_samples)
                indexes = np.array(list(range(num_samples)))
                for i in range(self.num_epochs):
                    np.random.shuffle(indexes)
                    index_batch = indexes[0:batch_size]
                    states_batch = states[index_batch]
                    accumulated_batch = accumulated_rewards[index_batch]
                    advantages_batch = np.zeros((batch_size, self.action_size))
                    single_dimension_advantages = accumulated_batch - self.brain.predict_values(states_batch)
                    for idx in range(batch_size):
                        advantages_batch[idx, transitions[index_batch[idx]]['a']] = single_dimension_advantages[idx]
                    old_policies_batch = old_policies[index_batch]
                    self.brain.train_network_batch(states_batch, advantages_batch, accumulated_batch, old_policies_batch)
                self.trajectories = []
