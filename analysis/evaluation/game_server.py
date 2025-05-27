from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from utils.global_functions import save_json
from .llm_settings import get_llm_client
from typing import List, Dict, Optional
import uuid
import random
from games.base_game import GameBase
from pydantic import BaseModel
from typing import Any
import random
import time

class GameState(BaseModel):
    game: Any
    meta: Dict[str, Any]
    player_data: Dict[str, List[Dict]]
    prompts: List[Dict[str, Any]] = []
    responses: List[Dict[str, Any]] = []

class GameServer:
    def __init__(self, game: "GameBase", model_config: List[Dict[str, Optional[str]]], player_num: int, default_model: Optional[Dict[str, Optional[str]]] = None):
        self.game = game
        self.player_num = player_num
        self.model_config = {f"player_{i}": model_config[i] for i in range(player_num)}
        self.default_model = default_model
        self.state = GameState(
            game=game,
            meta={"game_name": game.__class__.__name__, "model_config": self.model_config, "player_num": player_num},
            player_data={f"player_{i}": [] for i in range(player_num)}
        )
        self.graph = self._build_graph()
        # Seed random with current time to ensure different results each run
        random.seed(time.time())

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(GameState)

        def generate_prompts(state: GameState) -> GameState:
            round_num = len(state.player_data["player_0"]) if state.player_data["player_0"] else 0
            state.prompts = []
            for i in range(self.player_num):
                player_id = f"player_{i}"
                history = state.player_data.get(player_id, [])
                prompt_text = self.game.get_prompt(player_id, round_num, history)
                card = history[-1]["card"] if history else self.game.deck[i % len(self.game.deck)]
                state.prompts.append({"player_id": player_id, "prompt": prompt_text, "card": card})
            return state

        def collect_responses(state: GameState) -> GameState:
            state.responses = []
            for prompt_data in state.prompts:
                player_id = prompt_data["player_id"]
                prompt_text = prompt_data["prompt"]
                player_config = self.model_config.get(player_id, self.default_model)
                if player_config is None or not player_config.get("model"):
                    print(f"Warning: No model config for {player_id}. Skipping.")
                    state.responses.append({"player_id": player_id, "response": ""})
                    continue
                model = player_config.get("model")
                api_key = player_config.get("api_key")
                try:
                    # Ensure LLM client uses temperature for randomness
                    client, _ = get_llm_client(model, api_key, temperature=0.7)
                    prompt = ChatPromptTemplate.from_template(prompt_text)
                    chain = prompt | client
                    result = chain.invoke({})
                    action = result.content.strip()
                    if action not in ["Bet", "Pass"]:
                        raise ValueError(f"Invalid action: {action}")
                    player_index = int(player_id.split("_")[1])
                    state.responses.append({
                        "player_id": player_id,
                        "response": action,
                        "card": prompt_data.get("card", self.game.deck[player_index % len(self.game.deck)])
                    })
                except Exception as e:
                    print(f"Error for {player_id} with model {model}: {str(e)}")
                    state.responses.append({"player_id": player_id, "response": ""})
            return state

        def process_results(state: GameState) -> GameState:
            result = self.game.compute_result(state.responses)
            self.game.report_result(result, state.player_data)
            state.meta["result"] = result
            return state

        graph.add_node("generate_prompts", generate_prompts)
        graph.add_node("collect_responses", collect_responses)
        graph.add_node("process_results", process_results)

        graph.add_edge(START, "generate_prompts")
        graph.add_edge("generate_prompts", "collect_responses")
        graph.add_edge("collect_responses", "process_results")
        graph.add_edge("process_results", END)

        return graph.compile()

    def run(self, rounds: int, output_file: Optional[str] = None) -> List[Dict]:
        results = []
        # Reset player_data to ensure fresh start
        self.state.player_data = {f"player_{i}": [] for i in range(self.player_num)}
        for round_num in range(rounds):
            # Assign new random cards for each round
            cards = random.sample(self.game.deck, self.player_num)
            for i, player_id in enumerate(self.state.player_data):
                # Append new round data with card
                self.state.player_data[player_id].append({"card": cards[i], "state": ""})

            # Run one round
            result = self.graph.invoke(self.state)
            round_result = result.get("meta", {}).get("result", {})
            results.append({
                "round_id": round_num,
                "cards": round_result.get("cards", []),
                "actions": round_result.get("actions", []),
                "rewards": round_result.get("rewards", []),
                "player_data": {pid: data[-1] for pid, data in result.get("player_data", {}).items()}
            })

            # Reset prompts and responses for next round
            self.state.prompts = []
            self.state.responses = []

        if output_file:
            save_json(results, output_file)

        return results