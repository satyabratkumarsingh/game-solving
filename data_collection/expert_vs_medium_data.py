import os
import pyspiel
import random
from collect_utils import load_solver, save_final_file

NUM_GAMES = 5000

def simulate_games(game_name, solver1_path, solver2_path, num_games=10):
    game = pyspiel.load_game(game_name)

    solver1 = load_solver(solver1_path)
    solver2 = load_solver(solver2_path)

    policy1 = solver1.average_policy()
    policy2 = solver2.average_policy()

    players = [policy1, policy2]

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

            # Log the current state and player ID
            print(f"Episode {episode_id}, Current Player: {current_player}")
            print(f"State: {state}")

            # Handle chance nodes (current_player == -1)
            if current_player == -1:  # Chance node
                chance_outcomes = state.chance_outcomes()
                action, probability = random.choice(chance_outcomes)
                print(f"Chance node: sampled action {action} with probability {probability}")
                state.apply_action(action)
                continue

            # Ensure that current_player is valid
            if current_player < 0 or current_player >= game.num_players():
                print(f"Invalid current_player {current_player} at episode {episode_id}. Skipping game.")
                break

            try:
                # Ensure the player is valid before appending state information
                num_states.append(state.information_state_string(current_player))
                player_ids.append(current_player)

                # Get action probabilities for current player
                action_probs = players[current_player].action_probabilities(state)
                print(f"Action probabilities for player {current_player}: {action_probs}")

                if action_probs:
                    # Select action with highest probability
                    action = max(action_probs, key=action_probs.get)
                    print(f"Selected action for player {current_player}: {action}")

                    # Convert action to string
                    try:
                        action_string = state.action_to_string(current_player, action)
                        str_actions.append(action_string)
                        print(f"Action string: {action_string}")
                    except Exception as e:
                        print(f"Error converting action to string: {e}")
                        str_actions.append(None)
                else:
                    print(f"No action probabilities for player {current_player}")
                    str_actions.append(None)
                
                # Store the selected action
                num_actions.append(action)
                state.apply_action(action)
                str_states.append(str(state))
            except Exception as e:
                print(f"Error during simulation: {e}")
                break

        # Game Over, collect rewards
        rewards = state.returns()

        # Store the results for this game episode
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


def play_games():
    # Paths to solvers
    solver1_path = "../policies/kuhn_poker_solver_expert.pkl"
    solver2_path = "../policies/kuhn_poker_solver_medium.pkl"
    
    # Simulate the games
    results = simulate_games("kuhn_poker", solver1_path, solver2_path, num_games=NUM_GAMES)
    
    # Save the final results
    save_final_file("kuhn_poker", "mccfr", results, "expert_vs_expert")


def main():
    # Play the games and save the results
    play_games()
    

if __name__ == "__main__":
    main()
