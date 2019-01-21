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


class MOISMCTSWithValueNetworkAgent(Agent, MOIMCTSMixin):
    """
    MOISMCTSWithValueNetworkAgent class for playing with it
    """

    def __init__(self, iteration_count: int, state_size: int, num_players: int,
                 brain: ValueNetworkBrain = None, reuse_tree: bool = True, k: float = 0.2, gamma: float = 0.99):
        """
        Initializer for `MOISMCTSWithValueNetworkAgent`
        :param iteration_count: ???
        :param state_size: ???
        :param num_players: The number of players
        :param brain: ???
        :param reuse_tree: If we should reuse the tree or not
        :param k: ???
        :param gamma: ???
        """
        super(MOIMCTSMixin, self).__init__(k)
        self.iteration_count = iteration_count
        self.reuse_tree = reuse_tree
        self.gamma = gamma
        self.brain = brain
        self.num_players = num_players
        self.state_size = state_size
        if not brain:
            self.brain = ValueNetworkBrain(state_size, num_players)
        self.current_trajectory = []
        self.current_transition = None

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `MOISMCTSWithValueNetworkAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if not self.current_transition:
            return
        self.current_transition['r'] += reward
        self.current_transition['terminal'] |= terminal
        if terminal:
            R = 0
            self.current_trajectory.append(self.current_transition)
            self.current_transition = None
            for transition in reversed(self.current_trajectory):
                R = transition['r'] + self.gamma * R
                accumulated_rewards = np.ones(self.num_players) * R
                for i in range(self.num_players):
                    if i != transition['player_index']:
                        accumulated_rewards[i] = -R / (self.num_players - 1)
                transition['R'] = accumulated_rewards
            states = np.array([transition['s'] for transition in self.current_trajectory])
            target_values = np.array([transition['R'] for transition in self.current_trajectory])
            self.brain.train(states, target_values)
            self.current_trajectory = []

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `MOISMCTSWithValueNetworkAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        if self.current_transition:
            self.current_transition['terminal'] = False
            self.current_trajectory.append(self.current_transition)
            self.current_transition = None
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
            scores = self.brain.predict_state(info_state.vectorize())
            # BACKPROPAGATE SCORE
            for player_id in self.current_iteration_selected_nodes.keys():
                visited_nodes = self.current_iteration_selected_nodes[player_id]
                for node, child_action in reversed(visited_nodes):
                    node['nprime'] += 1
                    child_action['n'] += 1
                    child_action['r'] += scores[player_id]
        child_action = max(self.current_iteration_selected_nodes[player_index][0][0]['a'], key=lambda child: child['n'])
        self.current_transition = {
            's': information_state.vectorize(),
            'r': 0,
            'player_index': player_index,
            'terminal': False
        }
        return child_action['action_id']
