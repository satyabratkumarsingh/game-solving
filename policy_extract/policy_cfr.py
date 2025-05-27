import os
from open_spiel.python.algorithms import cfr
import pyspiel
import pickle
from  utils import calculate_ne_gap, calculate_exploitability, calculate_winning_rate


def save_policy_at_iteration(solver, iteration, game_name):
    policy_file = f"{game_name}_cfr_policy_iter_{iteration}.pkl"
    with open(policy_file, 'wb') as f:
        pickle.dump(solver, f)
    print(f"Policy saved at iteration {iteration} to {policy_file}")

# Train Expert Agent with periodic saving
def train_expert_with_saving(game_name, iterations=100000, save_freq=1000):
    game = pyspiel.load_game(game_name)
    solver = cfr.OutcomeSamplingSolver(game)

    for i in range(iterations):
        solver.iteration()

        # Save policy at specified iterations
        if i > 0 and i % save_freq == 0:
            save_policy_at_iteration(solver, i, game_name)

    return solver

