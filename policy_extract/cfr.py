from open_spiel.python.algorithms import cfr
from  utils import \
    calculate_ne_gap, calculate_exploitability, \
    calculate_winning_rate, save_solver, log_exploitability, \
    log_ne_gap, log_winning_rate
import pyspiel
import random
import json
import os

NUM_EPISODES = 5000
PRINT_FREQ = 10
SAVE_FREQ = 1000


def train_cfr(game_name):
    game = pyspiel.load_game(game_name)

    solver1 = cfr.CFRSolver(game)
    solver2 = cfr.CFRSolver(game)

    for ep in range(NUM_EPISODES):
        if ep % PRINT_FREQ == 0:
            print(f"Episode: {ep}")
            exploitability_value1 = calculate_exploitability(game, solver1)
            exploitability_value2 = calculate_exploitability(game, solver2)
            print(f"Exploitability For Solver 1: {exploitability_value1}")
            print(f"Exploitability For Solver 2: {exploitability_value2}")
            log_exploitability(game_name, ep, 'cfr', exploitability_value1)

        if ep > 0 and ep % SAVE_FREQ == 0:
            save_solver(game_name, solver1, ep, 1)
            save_solver(game_name, solver2, ep, 2)

        solver1.evaluate_and_update_policy()
        solver2.evaluate_and_update_policy()

    # Nash Equilibrium Gap
    ne_gap1 = calculate_ne_gap(game, solver1)
    ne_gap2 = calculate_ne_gap(game, solver2)
    
    log_ne_gap(game_name, ep, 'cfr', ne_gap1, ne_gap2)
    # Winning Rates
    winning_rates = calculate_winning_rate(game, solver1, solver2)

    log_winning_rate(game_name, ep, 'cfr', winning_rates[0], winning_rates[1])
    

    # save solver
    save_solver(game_name, solver1, NUM_EPISODES, 1)
    save_solver(game_name, solver2, NUM_EPISODES, 2)


def main():
    train_cfr("leduc_poker")
    train_cfr("kuhn_poker")


if __name__ == "__main__":
    main()
