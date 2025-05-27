import os
import json
import pickle
import random
import psutil
from tqdm import tqdm
import pyspiel

from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
from utils import log_evaluation_to_file, load_turn_based_game


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")


class RandomAgent:
    def __init__(self, game):
        self.game = game

    def step(self, time_step):
        return random.choice(time_step.legal_actions())


def eval_against_random(env, solver, num_episodes=1000):
    random_agent = RandomAgent(env)
    solver_policy = solver.average_policy()
    avg_rewards = []

    for player_id in range(env.num_players()):
        total_reward = 0
        for _ in range(num_episodes):
            state = env.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    actions, probs = zip(*state.chance_outcomes())
                    state.apply_action(random.choices(actions, probs)[0])
                else:
                    if state.current_player() == player_id:
                        action_probs = solver_policy.action_probabilities(state)
                        actions, probs = zip(*action_probs.items())
                        action = random.choices(actions, weights=probs)[0]
                    else:
                        action = random_agent.step(state)
                    state.apply_action(action)
            total_reward += state.returns()[player_id]
        avg_rewards.append(total_reward / num_episodes)

    print("Average reward against random:", avg_rewards)
    return avg_rewards


def eval_against_solver(env, solver, saved_solver, num_episodes=1000):
    solver_policy = solver.average_policy()
    saved_policy = saved_solver.average_policy()
    avg_rewards = []

    for player_id in range(env.num_players()):
        total_reward = 0
        for _ in range(num_episodes):
            state = env.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    actions, probs = zip(*state.chance_outcomes())
                    state.apply_action(random.choices(actions, probs)[0])
                else:
                    policy = solver_policy if state.current_player() == player_id else saved_policy
                    actions, probs = zip(*policy.action_probabilities(state).items())
                    state.apply_action(random.choices(actions, weights=probs)[0])
            total_reward += state.returns()[player_id]
        avg_rewards.append(total_reward / num_episodes)

    print("Average reward against medium:", avg_rewards)
    return avg_rewards


def load_saved_solver(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def main():
    game_name = 'oshi_zumo'
    iterations = 10000
    sampling = 'outcome'
    eval_episodes = 10000
    calculate_exploitability = False
    train_medium = False

    game = load_turn_based_game(game_name)
    print_freq = iterations // 20
    nash_conv_values = []

    solver = (external_mccfr.ExternalSamplingSolver(game, external_mccfr.AverageType.SIMPLE)
              if sampling == "external" else outcome_mccfr.OutcomeSamplingSolver(game))

    for i in tqdm(range(1, iterations)):
        solver.iteration()

        if i % print_freq == 0:
            if calculate_exploitability:
                conv = exploitability.nash_conv(game, solver.average_policy())
                print(f"Iteration {i} exploitability {conv}")
                nash_conv_values.append({"iteration": i, "nash_conv": conv})

            result = eval_against_random(game, solver, num_episodes=1000)
            log_evaluation_to_file(i, result, f"../log/eval_random_{game_name}_episode_{iterations}.csv")

            if not train_medium:
                saved_solver = load_saved_solver(f"../policies/{game_name}_solver_medium.pkl")
                result = eval_against_solver(game, solver, saved_solver, num_episodes=eval_episodes)
                log_evaluation_to_file(i, result, f"../log/eval_medium_{game_name}_episode_{iterations}.csv")

    with open(f'../policies/{game_name}_solver_expert_{iterations}.pkl', 'wb') as f:
        pickle.dump(solver, f)

    if calculate_exploitability:
        with open(f'../log/{game_name}_nash_conv_{iterations}.json', 'w') as f:
            json.dump(nash_conv_values, f, indent=4)


if __name__ == "__main__":
    main()
