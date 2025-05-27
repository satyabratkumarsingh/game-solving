# games/base_game.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Game(ABC):
    """
    Abstract base class for all games in GAMABench.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def run(self, agents: List[Any]) -> Dict[str, Any]:
        """
        Run one episode of the game with the given agents.

        Args:
            agents (List[Any]): List of agent objects, each with an `act(obs)` method.

        Returns:
            Dict[str, Any]: The result of the game, including history, rewards, etc.
        """
        pass

   