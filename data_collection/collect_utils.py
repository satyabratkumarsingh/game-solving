import os
import pickle
import json


def load_solver(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_final_file(game, algo, results, type):
    output_file = f"../data/{game}_{algo}_{type}_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_results(results, output_file)


def save_results(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
