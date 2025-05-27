import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Any
from utils.global_functions import load_json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Any
from utils.global_functions import load_json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Any
from utils.global_functions import load_json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Any
from utils.global_functions import load_json
import os

class Analysis:
    def __init__(self, game_class: Any):
        self.game_class = game_class
        self.results = {}  # Dict[str, List[Dict]]: {label: [result_dicts]}
        self.metrics = {}  # Dict[str, Dict]: {label: computed_metrics}

    def add(self, json_file: str, label: str) -> None:
        data = load_json(json_file)
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        self.results[label] = data
        self._compute_metrics(label)

    def _compute_metrics(self, label: str) -> None:
        data = self.results.get(label, [])
        if not data:
            self.metrics[label] = {}
            return

        action_counts = {f"player_{i}": {"Bet": 0, "Pass": 0} for i in range(len(data[0]["player_data"]))}
        rewards = {f"player_{i}": [] for i in range(len(data[0]["player_data"]))}
        scores = {f"player_{i}": [] for i in range(len(data[0]["player_data"]))}
        state_counts = {}
        action_pairs = []

        for result in data:
            actions = result.get("actions", []) if "actions" in result else list(result.get("predicted_actions", {}).values())
            rewards_list = result.get("rewards", [])
            player_data = result.get("player_data", {})
            for i, player_id in enumerate(player_data):
                action = actions[i] if i < len(actions) else ""
                if action in ["Bet", "Pass"]:
                    action_counts[player_id][action] += 1
                if i < len(rewards_list):
                    rewards[player_id].append(rewards_list[i])
                if "scores" in result:
                    scores[player_id].append(result["scores"].get(player_id, 0.0))
            if len(actions) >= 2:
                action_pairs.append((actions[0], actions[1]))
            if "player_1" in player_data and "state" in player_data["player_1"]:
                state = player_data["player_1"].get("state", "")
                state_counts[state] = state_counts.get(state, 0) + 1

        avg_rewards = {pid: np.mean(reward_list) if reward_list else 0.0 for pid, reward_list in rewards.items()}
        avg_scores = {pid: np.mean(score_list) if score_list else 0.0 for pid, score_list in scores.items()}
        pair_counts = {(a1, a2): action_pairs.count((a1, a2)) for a1, a2 in set(action_pairs)}

        self.metrics[label] = {
            "action_counts": action_counts,
            "avg_rewards": avg_rewards,
            "avg_scores": avg_scores,
            "state_counts": state_counts,
            "action_pairs": pair_counts,
            "rewards_per_round": {pid: [result["rewards"][i] for result in data] for i, pid in enumerate(player_data)}
        }

    def plot_comparison(self, models: List[str], mode: str = "online", savename: Optional[str] = None, figsize: tuple = (12, 5), font_scale: float = 1.2) -> None:
        """Plot a comparison of two models, separating players into subplots."""
        sns.set(style="whitegrid", font_scale=font_scale)

        # Determine the players (e.g., player_0, player_1)
        players = None
        for model in models:
            if model in self.results:
                players = list(self.metrics[model]["rewards_per_round"].keys()) if mode == "online" else list(self.metrics[model]["avg_scores"].keys())
                break
        if not players:
            print("No valid data to plot.")
            return

        # Create subplots: one for each player
        fig, axes = plt.subplots(1, len(players), figsize=figsize, sharey=True)
        if len(players) == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot

        if mode == "online":
            # Line Plot: Rewards Over Rounds for Each Player
            for idx, (player_id, ax) in enumerate(zip(players, axes)):
                for model in models:
                    if model not in self.results:
                        continue
                    rewards = self.metrics[model]["rewards_per_round"].get(player_id, [])
                    rounds = list(range(len(rewards)))
                    ax.plot(
                        rounds, 
                        rewards, 
                        marker="o", 
                        linestyle="-", 
                        linewidth=2, 
                        markersize=8, 
                        label=model,
                        color="#1f77b4" if "Open AI" in model else "#ff7f0e"
                    )
                ax.set_xlabel("Round", fontsize=12)
                ax.set_ylabel("Reward", fontsize=12)
                ax.set_title(f"Player {player_id.split('_')[1]}", fontsize=14, pad=15)
                ax.legend(loc="best", fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.7)

        elif mode == "offline":
            # Bar Plot: Average Scores for Each Player
            for idx, (player_id, ax) in enumerate(zip(players, axes)):
                for i, model in enumerate(models):
                    if model not in self.results:
                        continue
                    avg_scores = self.metrics[model]["avg_scores"]
                    score = avg_scores.get(player_id, 0.0)
                    ax.bar(
                        model,
                        score,
                        label=model if idx == 0 else None,
                        alpha=0.7,
                        color="#1f77b4" if "Open AI" in model else "#ff7f0e",
                        edgecolor="black"
                    )
                ax.set_xlabel("Model", fontsize=12)
                ax.set_ylabel("Average Score", fontsize=12)
                ax.set_title(f"Player {player_id.split('_')[1]}", fontsize=14, pad=15)
                if idx == 0:
                    ax.legend(loc="best", fontsize=10)
                ax.set_ylim(0, 1.1)  # Scores are between 0 and 1
                ax.grid(True, linestyle="--", alpha=0.7, axis="y")

        plt.tight_layout()
        if savename:
            plt.savefig(f"{savename}_{mode}_comparison_separated.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot(self, xlabel: str = "Metric", loc: str = "best", savename: Optional[str] = None, figsize: tuple = (10, 6), font_scale: float = 1.2) -> None:
        # Existing plot method (unchanged, included for completeness)
        sns.set(style="whitegrid", font_scale=font_scale)
        # ... (rest of the plot method unchanged)