from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
from utils import log_evaluation_to_file
from utils import load_turn_based_game
from tqdm import tqdm
import pyspiel
import pickle
import random
import os
import json

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")

class RandomAgent:
    def __init__(self, game):
        self.game = game

    def step(self, time_step):
        legal_actions = time_step.legal_actions()
        return random.choice(legal_actions)
    
def eval_against_random(env, solver, num_episodes=1000):
    random_agent = RandomAgent(env)
    solver_policy = solver.average_policy()
    num_players = 2
    avg_rewards = []
    for i in range(num_players):
        total_reward = 0
        for _ in range(num_episodes):
            state = env.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes_with_probs = state.chance_outcomes()
                    actions_with_probs, probs = zip(*outcomes_with_probs)
                    chosen_action = random.choices(actions_with_probs, probs)[0]
                    state.apply_action(chosen_action)
                else:
                    if state.current_player() == i:
                        action_probs = solver_policy.action_probabilities(state)
                        legal_actions = list(action_probs.keys())
                        probabilities = list(action_probs.values())
                        action = random.choices(legal_actions, weights=probabilities, k=1)[0]
                    else:
                        action = random_agent.step(state)
                    state.apply_action(action)
                    
            returns = state.returns()
            total_reward += returns[i]
        avg_rewards.append(total_reward/num_episodes)
    print("average reward against random", avg_rewards)
    return avg_rewards

def eval_against_solver(env, solver, saved_solver, num_episodes=1000):
    """ evaluate trained solver against a saved solver """
    solver_policy = solver.average_policy()
    saved_policy = saved_solver.average_policy()
    num_players = 2
    avg_rewards = []
    for i in range(num_players):
        total_reward = 0
        for _ in range(num_episodes):
            state = env.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes_with_probs = state.chance_outcomes()
                    actions_with_probs, probs = zip(*outcomes_with_probs)
                    chosen_action = random.choices(actions_with_probs, probs)[0]
                    state.apply_action(chosen_action)
                else:
                    if state.current_player() == i:
                        action_probs = solver_policy.action_probabilities(state)
                    else:
                        action_probs = saved_policy.action_probabilities(state)
                    legal_actions = list(action_probs.keys())
                    probabilities = list(action_probs.values())
                    action = random.choices(legal_actions, weights=probabilities, k=1)[0]
                    state.apply_action(action)
            returns = state.returns()
            total_reward += returns[i]
        avg_rewards.append(total_reward/num_episodes)
    print("average reward against medium", avg_rewards)
    return avg_rewards

def load_saved_solver(file_path):
    """Load a previously saved solver from a pickle file."""
    # Open the file and load the solver object
    with open(file_path, 'rb') as f:
        saved_solver = pickle.load(f)
        return saved_solver

def main():
    # args
    game_name = 'oshi_zumo'
    players = 2
    iterations = 10000
    print_freq = iterations//20
    sampling = 'outcome'
    eval_episodes = 10000
    calculate_exploitability = False # whether to calculate exploitability during training
    train_medium = False # whether to train a medium solver

    # transform into turn-based game if necessary
    game = load_turn_based_game(game_name)
    
    nash_conv_values = []
    
    # external or outcome sampling
    if sampling == "external":
        cfr_solver = external_mccfr.ExternalSamplingSolver(
            game, external_mccfr.AverageType.SIMPLE)
    else:
        cfr_solver = outcome_mccfr.OutcomeSamplingSolver(game)

    for i in tqdm(range(1, iterations)):
        cfr_solver.iteration()
        if i % print_freq == 0:
            if calculate_exploitability:
                # eval exploitability
                conv = exploitability.nash_conv(game, cfr_solver.average_policy())
                print("Iteration {} exploitability {}".format(i, conv))
                nash_conv_values.append({"iteration": i, "nash_conv": conv})
            # eval against random bots
            result = eval_against_random(game, cfr_solver, num_episodes=1000)
            log_evaluation_to_file(i, result, f"../log/eval_random_{game_name}_episode_{iterations}.csv")
            if not train_medium:
                # eval against medium bots
                saved_solver = load_saved_solver(f"../policy/{game_name}_solver_medium.pkl")
                result = eval_against_solver(game, cfr_solver, saved_solver, num_episodes=eval_episodes)
                log_evaluation_to_file(i, result, f"../log/eval_medium_{game_name}_episode_{iterations}.csv")
    
    with open(f'../policy/{game_name}_solver_expert_{iterations}.pkl', 'wb') as f:
        pickle.dump(cfr_solver, f)

    if calculate_exploitability:
        with open(f'../log/{game_name}_nash_conv_{iterations}.json', 'w') as f:
            json.dump(nash_conv_values, f, indent=4)

if __name__ == "__main__":
    main()
