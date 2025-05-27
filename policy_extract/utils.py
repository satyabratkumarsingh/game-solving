import numpy as np
import csv
import pyspiel
import random
from open_spiel.python.algorithms.exploitability import exploitability

def calculate_ne_gap(game, solver):
    """Calculate Nash Equilibrium Gap."""
    average_policy = solver.average_policy()
    ne_gap = exploitability.nash_conv(game, average_policy)
    return ne_gap

def calculate_exploitability(game, solver):
    """Calculate exploitability of the current policy."""
    average_policy = solver.average_policy()
    exploitability_value = exploitability.exploitability(game, average_policy)
    return exploitability_value

def calculate_winning_rate(game, solver1, solver2, num_simulations=1000):
    """Simulate games and calculate winning rate for both players."""
    wins = [0, 0]
    for _ in range(num_simulations):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes_with_probs = state.chance_outcomes()
                actions, probs = zip(*outcomes_with_probs)
                chosen_action = random.choices(actions, probs)[0]
                state.apply_action(chosen_action)
            else:
                current_player = state.current_player()
                solver = solver1 if current_player == 0 else solver2
                policy = solver.average_policy()
                action_probs = policy.action_probabilities(state)

                legal_actions = list(action_probs.keys())
                probabilities = list(action_probs.values())

                chosen_action = random.choices(legal_actions, weights=probabilities, k=1)[0]
                state.apply_action(chosen_action)
        final_returns = state.returns()
        if final_returns[0] > final_returns[1]:
            wins[0] += 1
        elif final_returns[1] > final_returns[0]:
            wins[1] += 1

    return [w / num_simulations for w in wins]

def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    num_players = len(trained_agents)
    sum_episode_rewards = np.zeros(num_players)
    for player_pos in range(num_players):
        cur_agents = random_agents[:]
        cur_agents[player_pos] = trained_agents[player_pos]
        for _ in range(num_episodes):
            time_step = env.reset()
            episode_rewards = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if env.is_turn_based:
                    agent_output = cur_agents[player_id].step(
                        time_step, is_evaluation=True)
                    action_list = [agent_output.action]
                else:
                    agents_output = [
                        agent.step(time_step, is_evaluation=True) for agent in cur_agents
                    ]
                    action_list = [agent_output.action for agent_output in agents_output]
                time_step = env.step(action_list)
                episode_rewards += time_step.rewards[player_pos]
            sum_episode_rewards[player_pos] += episode_rewards
    return sum_episode_rewards / num_episodes

def log_evaluation_to_file(episode, metric_value, filename="evaluation_log.csv"):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        # If the file is empty, write the header
        if file.tell() == 0:
            writer.writerow(["Episode", "Metric_Value"])
        writer.writerow([episode, metric_value])

def load_turn_based_game(game_name):
    '''load a turn based game, transform into turn-based game if necessary'''
    if game_name == 'goofspiel':
        game = pyspiel.load_game("turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=4,points_order=descending))")
    elif game_name == 'oshi_zumo':
        game = pyspiel.load_game("turn_based_simultaneous_game(game=oshi_zumo(coins=10))")
        print('============= HERE ##########')
    else:
        game = pyspiel.load_game(game_name)
    return game
