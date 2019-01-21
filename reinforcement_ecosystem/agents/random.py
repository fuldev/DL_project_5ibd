#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining Random Agent
"""


import random
from typing import Iterable

import numpy as np

from reinforcement_ecosystem.environments import InformationState, Agent, GameRunner


class RandomAgent(Agent):
    """
    Random Agent class for playing with it
    """

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `RandomAgent` does nothing
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        pass

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        action_count = len(available_actions)
        return available_actions[random.randint(0, action_count - 1)]


