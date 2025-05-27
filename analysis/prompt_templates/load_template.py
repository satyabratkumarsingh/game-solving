def load_template(path):
    with open(path, "r") as file:
        return file.read()

def generate_prompt_from_episode(episode, template_str):
    # Extract relevant info
    str_states = episode["str_states"]
    str_actions = episode["str_actions"]
    player_ids = episode["player_ids"]

    current_player = 1 - player_ids[-1]

    state_parts = str_states[0].split()
    if len(state_parts) < 2:
        raise ValueError("Malformed state: expected two cards and action string")
    player_card = state_parts[current_player]

    # Build action history string
    action_history = "\n".join([
        f"Player {pid}: {act}" for pid, act in zip(player_ids, str_actions)
    ])

    # Fill in template
    filled_prompt = template_str.format(
        player_id=current_player,
        player_card=player_card,
        action_history=action_history
    )

    return filled_prompt

