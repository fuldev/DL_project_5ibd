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


class MOIMCTSMixin:
    """
    MOISCMCTS Mixin for
    """

    def __init__(self, k: float = 0.2) -> None:
        """
        Initializer for the Mixin
        """
        self.current_iteration_selected_nodes = {}
        self.current_trees = {}
        self.k = k

    def add_visited_node(self, node: dict, selected_action: int, current_player: int) -> None:
        """
        Add the visited node to the current iteration  for a given player and action ???
        :param node: The node to be added
        :param selected_action: The selected action
        :param current_player: The current player ID
        """
        if current_player not in self.current_iteration_selected_nodes:
            self.current_iteration_selected_nodes[current_player] = []
        self.current_iteration_selected_nodes[current_player].append((node, selected_action))

    def select(self, gs: GameState) -> Tuple:
        """
        ???
        :param gs: The Game state to select on ???
        """
        terminal = False
        while True:
            current_player = gs.get_current_player_id()
            info_state = gs.get_information_state_for_player(current_player)
            if terminal:
                return gs, info_state, current_player, True
            if current_player not in self.current_trees:
                self.current_trees[current_player] = {}
            current_tree = self.current_trees[current_player]
            if info_state not in current_tree:
                current_tree[info_state] = {'nprime': 0}
                return gs, info_state, current_player, False
            current_node = current_tree[info_state]
            child_action = max(current_node['a'],
                               key=lambda node:
                               ((node['r'] / node['n'] + self.k * sqrt(log(current_node['nprime']) / node['n']))
                                   if node['n'] > 0 else 99999999))
            action_to_execute = child_action['action_id']
            self.add_visited_node(current_node, child_action, current_player)
            gs, reward, terminal = gs.step(current_player, action_to_execute)
