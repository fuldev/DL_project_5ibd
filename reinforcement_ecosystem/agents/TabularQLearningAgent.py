#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
from typing import Iterable

import numpy as np
import keras
from keras import Input, Model
from keras.activations import relu, linear
from keras.constraints import maxnorm
from keras.layers import Dense, Concatenate
from keras.optimizers import rmsprop

from reinforcement_ecosystem.environments import InformationState, Agent

class TabularQLearningAgent(Agent):
    """
    Tabular QLearning Agent class for playing with it
    """

    def __init__(self):
        """
        Initializer of the `TabularQLearningAgent`
        """
        self.Q = dict()
        self.s = None
        self.a = None
        self.r = None
        self.t = None
        self.s_next = None
        self.game_count = 0
        self.reward_history = np.array((0, 0, 0))

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `TabularQLearningAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if self.s is not None:
            self.r = (self.r if self.r else 0) + reward
            self.t = terminal
            if terminal:
                self.reward_history += (1 if reward == 1 else 0, 1 if reward == -1 else 0, 1 if reward == 0 else 0)
                self.learn()
                self.s = None
                self.a = None
                self.r = None
                self.t = None
                self.game_count += 1
                if (self.game_count % 1000) == 0:
                    print(self.reward_history / 1000)
                    self.reward_history = np.array((0, 0, 0))

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `TabularQLearningAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        if not (information_state in self.Q):
            self.Q[information_state] = dict()
            for action in available_actions:
                self.Q[information_state][action] = 1.1
        if self.s is not None:
            self.s_next = information_state
            self.learn()
        best_action = None
        best_action_score = 0
        for action in available_actions:
            if best_action is None or best_action_score < self.Q[information_state][action]:
                best_action = action
                best_action_score = self.Q[information_state][action]
        self.s = information_state
        self.a = best_action
        return best_action

    def learn(self):
        """
        Make the Agent (model) learn
        """
        self.Q[self.s][self.a] += 0.1 * (_reward_scaler(self.r) +
                                        (0 if self.t else (0.9 * max(self.Q[self.s_next].values()))) -
                                        self.Q[self.s][self.a])


def _reward_scaler(reward: int) -> float:
    """
    Scale a reward for a model
    :param reward: The reward to be scaled
    :return: The scaled reward
    """
    if reward == 1:
        return 1.0
    if reward == 0:
        return 0.0
    return -1.0
