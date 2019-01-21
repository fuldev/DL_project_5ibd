import os
from time import sleep, time

from agents.CommandLineAgent import CommandLineAgent
from agents.DeepQLearningAgent import DeepQLearningAgent
from agents.DoubleDeepQLearningAgent import DoubleDeepQLearningAgent
from agents.ReinforceClassicAgent import ReinforceClassicAgent
from agents.ReinforceClassicWithMultipleTrajectoriesAgent import ReinforceClassicWithMultipleTrajectoriesAgent
from agents.TabularQLearningAgent import TabularQLearningAgent
from agents.RandomAgent import RandomAgent
from agents.PPOWithMultipleTrajectoriesMultiOutputsAgent import PPOWithMultipleTrajectoriesMultiOutputsAgent
from agents.MOISMCTSWithRandomRolloutsAgent import MOISMCTSWithRandomRolloutsAgent
from agents.MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent import MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent
from agents.MOISMCTSWithValueNetworkAgent import MOISMCTSWithValueNetworkAgent
from games.tictactoe.runners.SafeTicTacToeRunner import SafeTicTacToeRunner

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from environments import Agent
from environments.GameRunner import GameRunner
from games.tictactoe.TicTacToeGameState import TicTacToeGameState
import numpy as np


class BasicTicTacToeRunner(GameRunner):

    def __init__(self, agent1: Agent, agent2: Agent,
                 print_and_reset_score_history_threshold=None,
                 replace_player1_with_commandline_after_similar_results=None,
                 file=None):

        self.agents = (agent1, agent2)
        self.stuck_on_same_score = 0
        self.prev_history = None
        self.print_and_reset_score_history_threshold = print_and_reset_score_history_threshold
        self.replace_player1_with_commandline_after_similar_results = replace_player1_with_commandline_after_similar_results
        self.execution_time = np.array((0.0, 0.0))
        self.file = file

    def run(self, max_rounds: int = -1,
            initial_game_state: TicTacToeGameState = TicTacToeGameState()) -> 'Tuple[float]':
        round_id = 0

        score_history = np.array((0, 0, 0))
        while round_id < max_rounds or round_id == -1:
            gs = initial_game_state.copy_game_state()
            terminal = False
            tour = 0
            execution_time = np.array((0.0, 0.0))
            while not terminal:
                current_player = gs.get_current_player_id()
                action_ids = gs.get_available_actions_id_for_player(current_player)
                info_state = gs.get_information_state_for_player(current_player)
                begin = time()
                action = self.agents[current_player].act(current_player,
                                                         info_state,
                                                         action_ids)
                end = time()
                # WARNING : Two Players Zero Sum Game Hypothesis
                (gs, score, terminal) = gs.step(current_player, action)
                self.agents[current_player].observe(
                    (1 if current_player == 0 else -1) * score,
                    terminal)

                execution_time[current_player] += end - begin

                if terminal:
                    score_history += (1 if score == 1 else 0, 1 if score == -1 else 0, 1 if score == 0 else 0)
                    other_player = (current_player + 1) % 2
                    self.agents[other_player].observe(
                        (1 if other_player == 0 else -1) * score,
                        terminal)

            # self.execution_time += execution_time / (tour * 0.5)

            if round_id != -1:
                round_id += 1
                if self.print_and_reset_score_history_threshold is not None and \
                        round_id % self.print_and_reset_score_history_threshold == 0:
                    print(score_history / self.print_and_reset_score_history_threshold)
                    if self.file is not None:
                        score_to_print = score_history / self.print_and_reset_score_history_threshold
                        execution_time_to_print = self.execution_time/self.print_and_reset_score_history_threshold
                        self.file.write(str(score_to_print[0]) + ";" + str(execution_time_to_print[0]) + ";"
                                        + str(score_to_print[1]) + ";" + str(execution_time_to_print[1]) + ";"
                                        + str(score_to_print[2]) + "\n")
                    if self.prev_history is not None and \
                            score_history[0] == self.prev_history[0] and \
                            score_history[1] == self.prev_history[1] and \
                            score_history[2] == self.prev_history[2]:
                        self.stuck_on_same_score += 1
                    else:
                        self.prev_history = score_history
                        self.stuck_on_same_score = 0
                    if (self.replace_player1_with_commandline_after_similar_results is not None and
                            self.stuck_on_same_score >= self.replace_player1_with_commandline_after_similar_results):
                        self.agents = (CommandLineAgent(), self.agents[1])
                        self.stuck_on_same_score = 0
                    score_history = np.array((0, 0, 0))
                    self.execution_time = np.array((0.0, 0.0))
        return tuple(score_history)


if __name__ == "__main__":

    number = [1000, 10000, 100000, 1000000]
    versus_name = ['RandomAgent', 'Tabular', 'DQN', 'DDQN', 'Reinforce', 'Reinforce A2C Style', 'PPO', 'MCTS']
    versus_agent = [RandomAgent(),
                    TabularQLearningAgent(),
                    DeepQLearningAgent(9, 9),
                    DoubleDeepQLearningAgent(9, 9),
                    ReinforceClassicAgent(9, 9),
                    ReinforceClassicWithMultipleTrajectoriesAgent(9, 9),
                    PPOWithMultipleTrajectoriesMultiOutputsAgent(9, 9),
                    MOISMCTSWithValueNetworkAgent(9, 9, 2)]
    versus = [versus_name, versus_agent]

	
    for num in number:
        for i in range(len(versus_name)):
            with open("D:/DEEP_LEARNING/Reinforcement/TabularVS" + str(versus[0][i]) +"_NB_"+ str(num) + ".csv", 'w+') as f: #Ici change TabularVS par le nom de l'agent que tu lance contre tout le reste
                print("New Fight" + str(versus[0][i]) + " " + str(num))
                begin = time()
                f.write("scoreJ1;execJ1;scoreJ2;execJ2;scoreEqual\n")
                print(BasicTicTacToeRunner(TabularQLearningAgent(), #Ici tu remplace TabularQLearningAgent() par un autre agent (ex : DeepQLearningAgent(9,9)
                                             versus[1][i],
                                             file=f).run(max_rounds=num))
                end = time()
                print(end - begin)
                f.write(str(end - begin) + "\n")
