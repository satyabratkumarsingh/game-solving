# generic_langgraph_game.py

import json
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from llm import get_llm


def load_prompt_template(template_path):
    with open(template_path, 'r') as f:
        template = f.read()
    return template


def substitute_template(template, input_data):
    """Replace placeholders in the template with input_data values."""
    for key, val in input_data.items():
        template = template.replace(f"!<{key}>!", str(val))
    return template


def create_player_node(model_name, template_path):
    llm = get_llm(model_name)
    template = load_prompt_template(template_path)

    def node_fn(state):
        player_id = state["current_player"]
        input_data = {
            "PLAYER_ID": player_id,
            "HISTORY": json.dumps(state.get("history", [])),
            **state.get("custom_inputs", {})
        }
        msg = substitute_template(template, input_data)
        result = llm.invoke([HumanMessage(content=msg)])

        action = result.content.strip().lower()
        state["history"].append((player_id, action))
        state["current_player"] = (player_id + 1) % len(state["players"])
        return state

    return node_fn


def build_generic_game_graph(model_names, prompt_template_path):
    builder = StateGraph()
    player_keys = [f"player_{i}" for i in range(len(model_names))]

    for i, model in enumerate(model_names):
        builder.add_node(player_keys[i], create_player_node(model, prompt_template_path))

    def router(state):
        if len(state["history"]) >= state["max_turns"]:
            return END
        return player_keys[state["current_player"]]

    builder.set_entry_point(player_keys[0])
    for player in player_keys:
        builder.add_conditional_edges(player, router)

    return builder.compile()


# Example usage
if __name__ == "__main__":
    game_state = {
        "history": [],
        "players": ["gpt-4", "meta-llama/Llama-3-8b-chat-hf"],
        "current_player": 0,
        "max_turns": 4,
        "custom_inputs": {
            "GAME_NAME": "Generic Strategy Game",
            "ROUND": 1
        }
    }

    graph = build_generic_game_graph(game_state["players"], "prompt_template/generic_game_template.txt")
    final_state = graph.invoke(game_state)
    print("Final History:", final_state["history"])
