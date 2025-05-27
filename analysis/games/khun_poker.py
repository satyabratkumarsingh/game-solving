from abc import ABC
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from utils.global_functions import read_prompt_template
from games.base_game import GameBase
from typing import List, Dict, Any


from abc import ABC
from typing import List, Dict, Any
from utils.global_functions import read_prompt_template
from games.base_game import GameBase

class KuhnPoker(GameBase):
    def __init__(self, deck: List[int] = [1, 2, 3], pot: float = 2.0):
        self.deck = deck
        self.pot = pot
        self.prompt_template = read_prompt_template("./prompt_templates/kuhn_poker_request.txt")

    def get_prompt(self, player_id: str, round: int, history: List[Dict]) -> str:
        card = history[-1]["card"] if history else self.deck[round % len(self.deck)]
        state_history = [h.get("state", "") for h in history if h.get("state") and h.get("state") != ""]
        prompt = self.prompt_template.format(
            player_id=player_id,
            card=card,
            state_history=state_history if state_history else ["None"]
        )
        print(f"Prompt for {player_id} (round {round}): {prompt}")  # Debugging
        return prompt

    def compute_result(self, responses: List[Dict]) -> Dict:
        actions = [r["response"] for r in responses]
        cards = [r.get("card", self.deck[i % len(self.deck)]) for i, r in enumerate(responses)]
        print(f"Computing result - Actions: {actions}, Cards: {cards}")  # Debugging
        if actions == ["Pass", "Pass"]:
            winner = 0 if cards[0] > cards[1] else 1
            rewards = [self.pot if i == winner else -1.0 for i in range(2)]
        elif actions == ["Pass", "Bet"]:
            rewards = [-1.0, 1.0]
        elif actions == ["Bet", "Pass"]:
            rewards = [1.0, -1.0]
        elif actions == ["Bet", "Bet"]:
            winner = 0 if cards[0] > cards[1] else 1
            rewards = [self.pot if i == winner else -self.pot for i in range(2)]
        else:
            rewards = [0.0, 0.0]  # Fallback for invalid actions
        print(f"Computed rewards: {rewards}")  # Debugging
        return {"cards": cards, "actions": actions, "rewards": rewards}

    def report_result(self, result: Dict, player_data: Dict[str, List]):
        feedback = f"Round result: Cards {result['cards']}, Actions {result['actions']}, Rewards {result['rewards']}"
        for i, player_id in enumerate(player_data):
            player_data[player_id][-1].update({
                "action": result["actions"][i],
                "feedback": feedback,
                "card": result["cards"][i],
                "state": result["actions"][0] if player_id == "player_0" else f"{result['actions'][0]} {result['actions'][1]}"
            })
        print(f"Updated player_data for {player_id}: {player_data[player_id][-1]}")  # Debugging

    def compute_score(self, round_records: List[Dict]) -> float:
        total_reward = sum(sum(r["rewards"]) / len(r["rewards"]) for r in round_records) / len(round_records)
        return max(0, min(100, (total_reward + self.pot) / (2 * self.pot) * 100))

    def parse_episode(self, episode: Dict) -> Dict:
        parsed = {}
        states = episode.get("str_states", [])
        state_history = states[:-1] if states else []
        for i, player_id in enumerate(episode.get("player_ids", [])):
            pid = f"player_{player_id}"
            try:
                card = int(episode["num_states"][i].replace("p", ""))
            except (KeyError, ValueError):
                card = self.deck[0]
            parsed[pid] = {
                "card": card,
                "state_history": state_history if state_history else ["None"],
                "ground_truth_action": episode.get("str_actions", [""])[i],
                "ground_truth_reward": episode.get("rewards", [0.0])[i]
            }
        return parsed

    def compute_offline_score(
        self, episode: Dict, parsed_data: Dict, predicted_action: Any, player_id: str
    ) -> float:
        ground_truth = parsed_data.get(player_id, {}).get("ground_truth_action", "")
        return 1.0 if predicted_action == ground_truth else 0.0

    def get_offline_prompt(self, player_id: str, episode: Dict, parsed_data: Dict) -> str:
        state_data = parsed_data.get(player_id, {})
        try:
            return self.prompt_template.format(
                player_id=player_id,
                card=state_data.get("card", "unknown"),
                state_history=state_data.get("state_history", ["None"])
            )
        except KeyError as e:
            raise KeyError(f"Missing prompt placeholder: {str(e)}")