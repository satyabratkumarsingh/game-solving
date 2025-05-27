import os
import pickle
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
import pyspiel
from utils import load_turn_based_game  # If needed, modify to just use pyspiel.load_game
from tqdm import tqdm

def train_medium_solver(game_name='kuhn_poker', iterations=10000, sampling='outcome', save_path='../policies'):
    # Load game
    try:
        game = load_turn_based_game(game_name)
    except:
        game = pyspiel.load_game(game_name)

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Select solver
    if sampling == 'external':
        solver = external_mccfr.ExternalSamplingSolver(game, external_mccfr.AverageType.SIMPLE)
    else:
        solver = outcome_mccfr.OutcomeSamplingSolver(game)

    print(f"Training medium solver for '{game_name}' using {sampling}-sampling MCCFR...")
    for i in tqdm(range(1, iterations + 1)):
        solver.iteration()

    # Save the solver
    file_path = os.path.join(save_path, f"{game_name}_solver_expert.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(solver, f)

    print(f"Saved medium-level solver to: {file_path}")

if __name__ == "__main__":
    train_medium_solver()
