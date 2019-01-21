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


class MOISMCTSWithRandomRolloutsAgent(Agent, MOIMCTSMixin):
    """
    Deep MOISMCTS with random rollouts Agent class for playing with it
    """

    def __init__(self, iteration_count: int, runner: GameRunner, reuse_tree: bool = True, k: float = 0.2):
        """
        Initializer for the `MOISMCTSWithRandomRolloutsAgent` class
        :param iteration_count: ???
        :param runner: ???
        :param reuse_tree: If we should reuse an existing tree
        :param k: ???
        """
        super(MOIMCTSMixin, self).__init__(k)
        self.iteration_count = iteration_count
        self.reuse_tree = reuse_tree
        self.runner = runner

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent` does nothing
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
        for i in range(self.iteration_count):
            self.current_iteration_selected_nodes = {}
            gs = information_state.create_game_state_from_information_state()
            # SELECT
            gs, info_state, current_player, terminal = self.select(gs)
            if not terminal:
                # EXPAND
                node = self.current_trees[current_player][info_state]
                available_actions = gs.get_available_actions_id_for_player(current_player)
                node['a'] = [{'n': 0, 'r': 0, 'action_id': action_id} for action_id in available_actions]
                child_action = random.choice(node['a'])
                action_to_execute = child_action['action_id']
                self.add_visited_node(node, child_action, current_player)
                gs, reward, terminal = gs.step(current_player, action_to_execute)
            # EVALUATE
            scores = self.runner.run(initial_game_state=gs, max_rounds=1)
            # BACKPROPAGATE SCORE
            for player_id in self.current_iteration_selected_nodes.keys():
                visited_nodes = self.current_iteration_selected_nodes[player_id]
                for node, child_action in reversed(visited_nodes):
                    node['nprime'] += 1
                    child_action['n'] += 1
                    child_action['r'] += scores[player_id]
        child_action = max(self.current_iteration_selected_nodes[player_index][0][0]['a'],
                           key=lambda child: child['n'])
        return child_action['action_id']
