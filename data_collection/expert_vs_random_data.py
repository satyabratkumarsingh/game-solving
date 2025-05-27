import os
import pyspiel
import random
from collect_utils import load_solver, save_final_file

NUM_GAMES = 5000

def simulate_games(game_name, solver1_path, num_games=10):
  
    game = pyspiel.load_game(game_name)

    solver1 = load_solver(solver1_path)
    policy1 = solver1.average_policy()
    players = [policy1, None]
    gameplay_data = []

    for episode_id in range(num_games):
        state = game.new_initial_state()

        str_states = []  
        num_states = [] 
        player_ids = [] 
        str_actions = []
        num_actions = []
        rewards = [] 

        while not state.is_terminal():

            current_player = state.current_player()
            if current_player == -1:
                chance_outcomes = state.chance_outcomes()
                action, probability = random.choice(chance_outcomes)
                print(f"Chance node: sampled action {action} with probability {probability}")
                state.apply_action(action)
                continue

            # Get the current player's action
            if current_player == 0:  # Expert (Player 1)
                action_probs = players[current_player].action_probabilities(state)
                action = max(action_probs, key=action_probs.get)
            else:  # Random Agent (Player 2)
                legal_actions = state.legal_actions()
                action = random.choice(legal_actions)

            # Log action and state information
            num_states.append(state.information_state_string(current_player))
            player_ids.append(current_player)
            num_actions.append(action)

            try:
                action_string = state.action_to_string(current_player, action)
                str_actions.append(action_string)
            except Exception as e:
                str_actions.append(None)
                print(e)

            state.apply_action(action)
            str_states.append(str(state))
               
        # Game Over, collect rewards
        rewards = state.returns()

        gameplay_data.append({
            "episode_id": episode_id,
            "str_states": str_states,
            "num_states": num_states,
            "player_ids": player_ids,
            "str_actions": str_actions,
            "num_actions": num_actions,
            "rewards": rewards
        })
    return gameplay_data


def play_leduc_poker():
    # CFR 
    solver1_path = "../policy/leduc_poker_solver1_cfr_episode_5000.pkl"
    results = simulate_games("leduc_poker", solver1_path,
                             num_games=NUM_GAMES)
    save_final_file("leduc_poker", "cfr", results, "expert_vs_random")

    # MCCFR 
    solver1_path = "../policy/leduc_poker_solver1_mccfr_episode_5000.pkl"
    results = simulate_games("leduc_poker", solver1_path,
                             num_games=NUM_GAMES)
    save_final_file("leduc_poker", "mccfr", results, "expert_vs_random")


def play_kuhn_poker():
    # CFR
    solver1_path = "../policy/kuhn_poker_solver1_cfr_episode_5000.pkl"
    results = simulate_games("kuhn_poker", solver1_path,
                             num_games=NUM_GAMES)
    save_final_file("kuhn_poker", "cfr", results, "expert_vs_random")

    # MCCFR 
    solver1_path = "../policy/kuhn_poker_solver1_mccfr_episode_5000.pkl"
    results = simulate_games("kuhn_poker", solver1_path,
                             num_games=NUM_GAMES)
    save_final_file("kuhn_poker", "mccfr", results, "expert_vs_random")


def main():
    play_leduc_poker()
    play_kuhn_poker()
    

if __name__ == "__main__":
    main()