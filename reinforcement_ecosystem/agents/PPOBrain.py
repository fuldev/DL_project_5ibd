#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining PPO Agent
"""

from typing import Any, Iterable

import numpy as np
import tensorflow as tf
from keras.constraints import max_norm

from reinforcement_ecosystem.environments import InformationState, Agent

__all__ = ['PPOWithMultipleTrajectoriesMultiOutputsAgent']


class PPOBrain:
    def __init__(self, state_size: int, num_layers: int, num_neuron_per_layer: int, action_size: int,
                 clip_epsilon: float = 0.2, c1: int = 1, c2: int = 0, session: tf.Session = None) -> 'PPOBrain':
        """
        Initializer for the PPO brain
        :param state_size: ???
        :param num_layers: The number of layer for the Model
        :param num_neuron_per_layer: The number of neuron per layer for the model
        :param action_size: ???
        :param clip_epsilon: ???
        :param c1: ???
        :param c2: ??? # No entropy for now
        :param session: The Tensorflow Session to use
        """
        self.state_size = state_size
        self.num_layers = num_layers
        self.num_neuron_per_layer = num_neuron_per_layer
        self.action_size = action_size
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.states_ph = tf.placeholder(shape=(None, state_size), dtype=tf.float32)
        self.advantages_ph = tf.placeholder(dtype=tf.float32)
        self.accumulated_reward_ph = tf.placeholder(dtype=tf.float32)
        self.old_policies_ph = tf.placeholder(dtype=tf.float32)
        self.policy_output, self.value_output, self.train_op = self.create_model()
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()
        if not self.session:
            self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_model(self) -> tuple:
        """
        Create Model from the internal state of the class
        :return: ???
        """
        hidden_policy = self.states_ph
        for i in range(self.num_layers):
            hidden_policy = tf.layers.batch_normalization(hidden_policy)
            hidden_policy = tf.layers.dense(hidden_policy, self.num_neuron_per_layer,
                                            activation=tf.keras.activations.relu,
                                            kernel_constraint=max_norm(16),
                                            bias_constraint=max_norm(16))
        policy_output = (tf.layers.dense(hidden_policy, self.action_size,
                                         activation=tf.keras.activations.softmax,
                                         kernel_constraint=max_norm(16),
                                         bias_constraint=max_norm(16))
                         + 0.000001)  # for numeric stability
        hidden_value = self.states_ph
        for i in range(self.num_layers):
            hidden_value = tf.layers.batch_normalization(hidden_value)
            hidden_value = tf.layers.dense(hidden_value, self.num_neuron_per_layer,
                                           activation=tf.keras.activations.relu,
                                           kernel_constraint=max_norm(16),
                                           bias_constraint=max_norm(16))

        value_output = tf.squeeze(tf.layers.dense(hidden_value, 1, activation=tf.keras.activations.linear,
                                                  kernel_constraint=max_norm(16),
                                                  bias_constraint=max_norm(16)), 1)
        advantages = self.advantages_ph
        r = policy_output / self.old_policies_ph
        # Lclip
        policy_loss = -tf.minimum(tf.multiply(r, advantages),
                                  tf.multiply(tf.clip_by_value(r, 1 - self.clip_epsilon,
                                                               1 + self.clip_epsilon),
                                              advantages))
        value_loss = tf.reduce_mean(tf.square(value_output - self.accumulated_reward_ph))
        entropy_loss = -policy_output * tf.log(policy_output)
        full_loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
        train_op = tf.train.AdamOptimizer().minimize(full_loss)
        return policy_output, value_output, train_op

    def predict_policy(self, state: Any) -> Any:
        """
        ???
        :param state: ???
        :return: ???
        """
        return self.session.run(self.policy_output, feed_dict={
            self.states_ph: [state]
        })[0]

    def predict_policy_and_value(self, state: Any) -> Any:
        """
        ???
        :param state: ???
        :return: ???
        """
        pol, val = self.session.run([self.policy_output, self.value_output], feed_dict={
            self.states_ph: [state]
        })
        return pol[0], val[0]

    def predict_policies(self, states: Any) -> Any:
        """
        ???
        :param states: ???
        :return: ???
        """
        return self.session.run(self.policy_output, feed_dict={
            self.states_ph: states,
        })

    def predict_values(self, states: Any) -> Any:
        """
        ???
        :param states: ???
        :return: ???
        """
        return self.session.run(self.value_output, feed_dict={
            self.states_ph: states
        })

    def predict_policies_and_values(self, states: Any) -> Any:
        """
        ???
        :param states: ???
        :return: ???
        """
        return self.session.run([self.policy_output, self.value_output], feed_dict={
            self.states_ph: states
        })

    def train_network(self, state: Any, advantage: Any, accumulated_reward: Any, old_policy: Any) -> Any:
        """
        Train the PPO Brain model
        :param state: ???
        :param advantage: ???
        :param accumulated_reward: ???
        :param old_policy: ???
        :return: ???
        """
        return self.session.run(self.train_op, feed_dict={
            self.states_ph: [state],
            self.advantages_ph: [advantage],
            self.accumulated_reward_ph: [accumulated_reward],
            self.old_policies_ph: [old_policy]
        })

    def train_network_batch(self, states: Any, advantages: Any, accumulated_rewards: Any, old_policies: Any) -> Any:
        """
        Train the PPO Brain model in batch
        :param states: ???
        :param advantages: ???
        :param accumulated_rewards: ???
        :param old_policies: ???
        :return: ???
        """
        return self.session.run(self.train_op, feed_dict={
            self.states_ph: states,
            self.advantages_ph: advantages,
            self.accumulated_reward_ph: accumulated_rewards,
            self.old_policies_ph: old_policies
        })
