from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from utils.global_functions import parse_json_response
from utils.global_functions import load_json
from games.base_game import GameBase
from games.khun_poker import KuhnPoker
from .llm_settings import get_llm_client
from typing import Type, Dict, Optional, Any
from pydantic import BaseModel
from utils.global_functions import read_prompt_template

class OfflineEvalState(BaseModel):
    episode: Dict[str, Any]
    parsed_data: Dict[str, Any] = {}
    predicted_actions: Dict[str, Any] = {}
    scores: Dict[str, float] = {}
    game_name: str = ""

GAME_CLASSES = {
    "KuhnPoker": KuhnPoker
}

def build_offline_eval_graph(
    game_name: str, model_config: Dict[str, Dict[str, Optional[str]]], default_model: Optional[Dict[str, Optional[str]]] = None
) -> StateGraph:
    if game_name not in GAME_CLASSES:
        raise ValueError(f"Unknown game: {game_name}")
    game_instance: GameBase = GAME_CLASSES[game_name]()

    graph = StateGraph(OfflineEvalState)

    def parse_episode(state: OfflineEvalState) -> OfflineEvalState:
        state.game_name = game_name
        state.parsed_data = game_instance.parse_episode(state.episode)
        return state

    def decision_maker(state: OfflineEvalState) -> OfflineEvalState:
        """Prompt LLM to predict actions for each player using game-specific prompt."""
        for player_id in state.episode.get("player_ids", []):
            player_config = model_config.get(player_id, default_model)
            if player_config is None:
                print(f"Warning: No model config for {player_id}. Skipping action prediction.")
                state.predicted_actions[player_id] = ""
                continue

            model = player_config.get("model")
            api_key = player_config.get("api_key")
            if not model:
                print(f"Warning: No model specified for {player_id}. Skipping action prediction.")
                state.predicted_actions[player_id] = ""
                continue

            try:
                client, _ = get_llm_client(model, api_key)
                # Get game-specific prompt
                prompt_text = game_instance.get_offline_prompt(player_id, state.episode, state.parsed_data)
                prompt = ChatPromptTemplate.from_template(prompt_text)
                chain = prompt | client
                result = chain.invoke({})  # No additional variables needed
                state.predicted_actions[player_id] = parse_json_response(result.content).get("action", "")
            except Exception as e:
                print(f"Error for {player_id} with model {model}: {str(e)}")
                state.predicted_actions[player_id] = ""
        return state

    def score_computer(state: OfflineEvalState) -> OfflineEvalState:
        for player_id in state.predicted_actions:
            state.scores[player_id] = game_instance.compute_offline_score(
                state.episode, state.parsed_data, state.predicted_actions[player_id], player_id
            )
        return state

    graph.add_node("parse_episode", parse_episode)
    graph.add_node("decision_maker", decision_maker)
    graph.add_node("score_computer", score_computer)

    graph.add_edge(START, "parse_episode")
    graph.add_edge("parse_episode", "decision_maker")
    graph.add_edge("decision_maker", "score_computer")
    graph.add_edge("score_computer", END)

    return graph.compile()

def evaluate_offline(
    json_file: str, game_name: str, model_config: Dict[str, Dict[str, Optional[str]]], default_model: Optional[Dict[str, Optional[str]]] = None
) -> Dict:
    eval_graph = build_offline_eval_graph(game_name, model_config, default_model)
    json_data = load_json(json_file)
    episodes = json_data if isinstance(json_data, list) else json_data.get("episodes", [])
    
    results = []
    for ep in episodes:
        result = eval_graph.invoke(OfflineEvalState(episode=ep, game_name=game_name))
        results.append({
            "episode_id": ep.get("episode_id", -1),
            "predicted_actions": result.get("predicted_actions", {}),  # Access as dict
            "scores": result.get("scores", {}),  # Access as dict
            "ground_truth_actions": ep.get("str_actions", [])
        })
    
    avg_scores = {
        pid: sum(r["scores"].get(pid, 0) for r in results) / len(results)
        for pid in results[0]["scores"] if results
    }
    return {"results": results, "average_scores": avg_scores}