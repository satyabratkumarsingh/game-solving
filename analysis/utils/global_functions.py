import json
import random
import json
import os
import re
from langchain_core.tools import tool
from typing import Dict, Any, Optional


@tool
def parse_json_response(response: str) -> Dict:
    """Extract JSON from an LLM response string."""
    try:
        # Look for JSON-like content within the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            return {"error": "No JSON found in response"}
        json_str = json_match.group(0)
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in response"}
    
def load_json(filepath: str) -> Dict:
    """Load a JSON file and return its contents as a dictionary."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in {filepath}")

def save_json(data: Dict, filepath: str) -> None:
    """Save a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def read_prompt_template(template_path: str) -> str:
    """Read a prompt template from a text file."""
    try:
        with open(template_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template {template_path} not found")

def ratio_randomization(options: list, probabilities: list) -> Any:
    """Select an option based on given probabilities."""
    if len(options) != len(probabilities):
        raise ValueError("Options and probabilities must have the same length")
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("Probabilities must sum to 1")
    return random.choices(options, probabilities)[0]

def add_noise(value: float, noise_level: float = 0.1) -> float:
    """Add random noise to a numeric value."""
    return value + random.uniform(-noise_level, noise_level)

def assign_cards(deck: list, player_num: int) -> list:
    """Randomly assign cards to players from the deck without replacement."""
    if len(deck) < player_num:
        raise ValueError("Deck size must be at least equal to player number")
    return random.sample(deck, player_num)