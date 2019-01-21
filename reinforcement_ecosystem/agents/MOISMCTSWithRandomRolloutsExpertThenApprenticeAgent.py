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


class MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent(Agent, MOIMCTSMixin):
    """
    MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent class for playing with it
    """

    def __init__(self, iteration_count: int, runner: GameRunner, state_size, action_size, reuse_tree=True,
                 training_episodes=3000, evaluation_episodes=1000, k=0.2) -> None:
        """
        Initializer for the `MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent` class
        :param iteration_count: ???
        :param runner: ???
        :param state_size: ???
        :param action_size: ???
        :param reuse_tree: If we should reuse an existing tree or not
        :param training_episodes: ???
        :param evaluation_episodes: ???
        :param k: ???
        """
        super(MOIMCTSMixin, self).__init__(k)
        self.iteration_count = iteration_count
        self.reuse_tree = reuse_tree
        self.training_episodes = training_episodes
        self.evaluation_episodes = evaluation_episodes
        self.state_size = state_size
        self.action_size = action_size
        self.runner = runner
        self.current_episode = 0
        self.X = []
        self.Y = []
        self.model = self.create_model()

    def create_model(self) -> 'keras.models.Model':
        """
        Build frol ????
        :return: A build keras model
        """
        model = Sequential()
        model.add(Dense(512, activation=relu, input_dim=self.state_size))
        model.add(Dense(512, activation=relu))
        model.add(Dense(self.action_size, activation=softmax))
        model.compile(optimizer=Adam(), loss=categorical_crossentropy)
        return model

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if terminal:
            self.current_episode += 1
            if self.current_episode == self.training_episodes:
                self.model.fit(np.array(self.X), np.array(self.Y), 512, epochs=2048)

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        available_actions = list(available_actions)
        if self.evaluation_episodes + self.training_episodes < self.current_episode:
            self.X = []
            self.Y = []
            self.current_episode = 0
        if self.current_episode > self.training_episodes:
            probs = self.model.predict(np.array([information_state.vectorize()]))[0]
            available_probs = probs[np.array(available_actions)]
            probs_sum = np.sum(available_probs)
            if probs_sum > 0.001:
                chosen_action_index = np.argmax(available_probs)
                action = available_actions[chosen_action_index]
            else:
                action = random.choice(available_actions)
            return action
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
        child_action = max(self.current_iteration_selected_nodes[player_index][0][0]['a'], key=lambda child: child['n'])
        self.X.append(information_state.vectorize().tolist())
        self.Y.append(to_categorical(child_action['action_id'], self.action_size))
        return child_action['action_id']
