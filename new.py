import json
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser
from langgraph import StateGraph, END
import pandas as pd
from dataclasses import dataclass

# ===== DATA MODELS =====

class KuhnPokerEpisode(BaseModel):
    """Single episode from offline data"""
    episode_id: int
    str_states: List[str]
    num_states: List[str] 
    player_ids: List[int]
    str_actions: List[str]
    num_actions: List[int]
    rewards: List[float]

class EvaluationState(BaseModel):
    """State for LangGraph evaluation workflow"""
    episode: KuhnPokerEpisode
    current_decision_point: int = 0
    llm_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_scores: Dict[str, float] = Field(default_factory=dict)
    analysis_complete: bool = False
    error_message: Optional[str] = None

# ===== KUHN POKER GAME LOGIC =====

class KuhnPokerAnalyzer:
    """Analyzes Kuhn Poker game states and optimal actions"""
    
    def __init__(self):
        self.card_mapping = {"1": "Jack", "2": "Queen", "3": "King"}
        self.action_mapping = {"Pass": 0, "Bet": 1}
    
    def parse_state(self, str_state: str) -> Dict[str, Any]:
        """Parse string state like '2 1 p' into structured format"""
        parts = str_state.strip().split()
        
        if len(parts) < 2:
            return {"error": "Invalid state format"}
            
        try:
            my_card = int(parts[0])
            opponent_card = int(parts[1]) if parts[1].isdigit() else None
            
            # Parse action history (p = pass, b = bet)
            action_history = ""
            if len(parts) > 2:
                action_history = parts[2].lower()
            
            return {
                "my_card": my_card,
                "my_card_name": self.card_mapping.get(str(my_card), "Unknown"),
                "opponent_card": opponent_card,
                "action_history": action_history,
                "position": "first" if not action_history else "second",
                "valid_actions": ["Pass", "Bet"]
            }
        except Exception as e:
            return {"error": f"Failed to parse state: {e}"}
    
    def get_optimal_action(self, state_info: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal action based on game theory"""
        if "error" in state_info:
            return {"action": "Pass", "confidence": 0.0, "reasoning": "Invalid state"}
        
        my_card = state_info["my_card"]
        action_history = state_info["action_history"]
        position = state_info["position"]
        
        # Simplified optimal strategy for Kuhn Poker
        if my_card == 3:  # King - always bet/call
            optimal_action = "Bet"
            confidence = 1.0
            reasoning = "King is highest card - always aggressive"
        elif my_card == 1:  # Jack - always pass
            if action_history == "":  # First to act
                optimal_action = "Pass"
                confidence = 1.0
                reasoning = "Jack is lowest card - pass and fold to bet"
            else:  # Responding to bet
                optimal_action = "Pass"
                confidence = 1.0
                reasoning = "Jack is lowest card - fold to opponent's bet"
        else:  # Queen - mixed strategy
            if action_history == "":  # First to act
                optimal_action = "Pass"  # Simplified - usually check
                confidence = 0.7
                reasoning = "Queen - check and decide based on opponent's action"
            elif action_history == "p":  # Opponent passed
                optimal_action = "Bet"
                confidence = 0.8
                reasoning = "Queen after opponent check - bet for value"
            else:  # Opponent bet
                optimal_action = "Pass"
                confidence = 0.6
                reasoning = "Queen - fold to opponent's bet (simplified strategy)"
        
        return {
            "action": optimal_action,
            "confidence": confidence,
            "reasoning": reasoning,
            "card_strength": my_card
        }

# ===== LLM AGENT FOR POKER DECISIONS =====

class KuhnPokerAgent:
    """LLM agent that makes Kuhn Poker decisions"""
    
    def __init__(self, llm, analyzer: KuhnPokerAnalyzer):
        self.llm = llm
        self.analyzer = analyzer
        self.memory = ConversationBufferMemory()
        
        # Create tools for the agent
        self.tools = [
            Tool(
                name="analyze_poker_state",
                description="Analyze current poker state and get game context",
                func=self._analyze_state_tool
            ),
            Tool(
                name="get_optimal_strategy",
                description="Get theoretical optimal action for comparison",
                func=self._get_optimal_strategy_tool
            )
        ]
        
        # Load prompt from text file
        self.prompt_template = self._load_prompt_template()
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        
        # Create the agent
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from text file"""
        try:
            with open("kuhn_poker_prompt.txt", "r") as f:
                base_prompt = f.read().strip()
            
            # Add JSON response format to the base prompt
            full_prompt = base_prompt + """

Please respond in JSON format with your decision:
{
    "action": "Bet" or "Pass",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your decision"
}
"""
            return full_prompt
            
        except FileNotFoundError:
            # Fallback to default prompt if file not found
            return """You are playing Kuhn Poker. The deck has 3 cards: J, Q, and K, coded as 1, 2, 3.
Each player is dealt one card. You are Player {player_id}, and your card is {player_card}.

Previous actions:
{action_history}

Now it's your turn.
Available actions: [Bet, Pass]

What do you want to do?
Please respond in JSON format with your decision:
{{
    "action": "Bet" or "Pass",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your decision"
}}
"""
        """Tool to analyze poker state"""
        state_info = self.analyzer.parse_state(state_str)
        return json.dumps(state_info, indent=2)
    
    def _get_optimal_strategy_tool(self, state_str: str) -> str:
        """Tool to get optimal strategy"""
        state_info = self.analyzer.parse_state(state_str)
        optimal = self.analyzer.get_optimal_action(state_info)
        return json.dumps(optimal, indent=2)
    
    async def make_decision(self, state_str: str, player_id: int) -> Dict[str, Any]:
        """Make a decision for given poker state"""
        try:
            state_info = self.analyzer.parse_state(state_str)
            
            if "error" in state_info:
                return {
                    "action": "Pass",
                    "confidence": 0.0,
                    "reasoning": f"Error parsing state: {state_info['error']}"
                }
            
            # Format action history for prompt
            action_history = state_info["action_history"]
            if not action_history:
                action_history = "No previous actions"
            else:
                # Convert 'p' to 'Pass', 'b' to 'Bet' for readability
                formatted_history = action_history.replace('p', 'Pass').replace('b', 'Bet')
                action_history = f"Previous actions: {formatted_history}"
            
            # Prepare input for the agent using the loaded prompt template
            agent_input = {
                "input": f"Make decision for Player {player_id}",
                "player_id": player_id,
                "player_card": f"{state_info['my_card']} ({state_info['my_card_name']})",
                "action_history": action_history
            }
            
            # Get agent's decision
            result = await self.agent_executor.ainvoke(agent_input)
            
            # Parse the JSON response
            try:
                decision = json.loads(result["output"])
                return decision
            except json.JSONDecodeError:
                # Try to extract action from plain text if JSON parsing fails
                output_text = result["output"].upper()
                if "BET" in output_text:
                    action = "Bet"
                elif "PASS" in output_text:
                    action = "Pass"
                else:
                    action = "Pass"  # Default fallback
                
                return {
                    "action": action,
                    "confidence": 0.5,
                    "reasoning": result["output"]
                }
                
        except Exception as e:
            return {
                "action": "Pass",
                "confidence": 0.0,
                "reasoning": f"Error making decision: {str(e)}"
            }

# ===== LANGGRAPH EVALUATION WORKFLOW =====

class KuhnPokerEvaluator:
    """Main evaluator using LangGraph workflow"""
    
    def __init__(self, llm):
        self.llm = llm
        self.analyzer = KuhnPokerAnalyzer()
        self.agent = KuhnPokerAgent(llm, self.analyzer)
        self.workflow = self._create_evaluation_workflow()
    
    def _create_evaluation_workflow(self):
        """Create LangGraph workflow for evaluation"""
        
        # Define the workflow graph
        workflow = StateGraph(EvaluationState)
        
        # Add nodes
        workflow.add_node("parse_episode", self._parse_episode_node)
        workflow.add_node("make_llm_decision", self._make_llm_decision_node)
        workflow.add_node("evaluate_decision", self._evaluate_decision_node)
        workflow.add_node("compute_scores", self._compute_scores_node)
        workflow.add_node("finalize_analysis", self._finalize_analysis_node)
        
        # Define edges
        workflow.add_edge("parse_episode", "make_llm_decision")
        workflow.add_edge("make_llm_decision", "evaluate_decision")
        workflow.add_edge("evaluate_decision", "compute_scores")
        workflow.add_edge("compute_scores", "finalize_analysis")
        workflow.add_edge("finalize_analysis", END)
        
        # Set entry point
        workflow.set_entry_point("parse_episode")
        
        return workflow.compile()
    
    async def _parse_episode_node(self, state: EvaluationState) -> EvaluationState:
        """Parse episode data and prepare for evaluation"""
        try:
            episode = state.episode
            print(f"Processing episode {episode.episode_id}")
            
            # Validate episode data
            if not episode.str_states or not episode.str_actions:
                state.error_message = "Empty states or actions in episode"
                return state
            
            # Initialize decision points (each state where a decision was made)
            state.current_decision_point = 0
            
            print(f"Episode has {len(episode.str_states)} states and {len(episode.str_actions)} actions")
            return state
            
        except Exception as e:
            state.error_message = f"Error parsing episode: {str(e)}"
            return state
    
    async def _make_llm_decision_node(self, state: EvaluationState) -> EvaluationState:
        """Get LLM decision for each decision point"""
        try:
            episode = state.episode
            predictions = []
            
            # Evaluate each decision point
            for i, (str_state, actual_action) in enumerate(zip(episode.str_states, episode.str_actions)):
                print(f"Evaluating decision point {i}: state='{str_state}', actual_action='{actual_action}'")
                
                # Get player ID for this decision
                player_id = episode.player_ids[i] if i < len(episode.player_ids) else 0
                
                # Get LLM's decision
                llm_decision = await self.agent.make_decision(str_state, player_id)
                
                # Store prediction with context
                prediction = {
                    "decision_point": i,
                    "state": str_state,
                    "actual_action": actual_action,
                    "llm_decision": llm_decision,
                    "player_id": episode.player_ids[i],
                    "reward": episode.rewards[i]
                }
                predictions.append(prediction)
            
            state.llm_predictions = predictions
            return state
            
        except Exception as e:
            state.error_message = f"Error making LLM decisions: {str(e)}"
            return state
    
    async def _evaluate_decision_node(self, state: EvaluationState) -> EvaluationState:
        """Evaluate LLM decisions against actual actions and optimal strategy"""
        try:
            evaluations = []
            
            for pred in state.llm_predictions:
                # Get optimal action for comparison
                state_info = self.analyzer.parse_state(pred["state"])
                optimal = self.analyzer.get_optimal_action(state_info)
                
                # Evaluate accuracy
                llm_action = pred["llm_decision"]["action"]
                actual_action = pred["actual_action"]
                optimal_action = optimal["action"]
                
                evaluation = {
                    "decision_point": pred["decision_point"],
                    "action_accuracy": 1.0 if llm_action == actual_action else 0.0,
                    "optimal_accuracy": 1.0 if llm_action == optimal_action else 0.0,
                    "llm_confidence": pred["llm_decision"].get("confidence", 0.5),
                    "reward": pred["reward"],
                    "state_info": state_info,
                    "optimal_info": optimal
                }
                evaluations.append(evaluation)
            
            # Add evaluations to predictions
            for i, evaluation in enumerate(evaluations):
                state.llm_predictions[i]["evaluation"] = evaluation
            
            return state
            
        except Exception as e:
            state.error_message = f"Error evaluating decisions: {str(e)}"
            return state
    
    async def _compute_scores_node(self, state: EvaluationState) -> EvaluationState:
        """Compute aggregate scores for the episode"""
        try:
            if not state.llm_predictions:
                state.evaluation_scores = {"error": "No predictions to score"}
                return state
            
            # Extract evaluations
            evaluations = [pred["evaluation"] for pred in state.llm_predictions]
            
            # Compute aggregate metrics
            action_accuracies = [e["action_accuracy"] for e in evaluations]
            optimal_accuracies = [e["optimal_accuracy"] for e in evaluations]
            confidences = [e["llm_confidence"] for e in evaluations]
            rewards = [pred["reward"] for pred in state.llm_predictions]
            
            scores = {
                "action_accuracy": sum(action_accuracies) / len(action_accuracies) if action_accuracies else 0.0,
                "optimal_accuracy": sum(optimal_accuracies) / len(optimal_accuracies) if optimal_accuracies else 0.0,
                "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                "total_reward": sum(rewards),
                "episode_length": len(state.llm_predictions),
                "decisions_evaluated": len(evaluations)
            }
            
            state.evaluation_scores = scores
            return state
            
        except Exception as e:
            state.error_message = f"Error computing scores: {str(e)}"
            return state
    
    async def _finalize_analysis_node(self, state: EvaluationState) -> EvaluationState:
        """Finalize the analysis with summary insights"""
        try:
            # Generate summary using LLM
            summary_prompt = f"""
            Analyze this Kuhn Poker episode evaluation:
            
            Episode ID: {state.episode.episode_id}
            Scores: {state.evaluation_scores}
            
            Decision Details:
            {json.dumps([pred["llm_decision"] for pred in state.llm_predictions], indent=2)}
            
            Provide a brief analysis of:
            1. Overall performance assessment
            2. Key strengths and weaknesses
            3. Strategic insights
            
            Keep it concise (2-3 sentences per point).
            """
            
            analysis = await self.llm.ainvoke(summary_prompt)
            state.evaluation_scores["llm_analysis"] = analysis.content
            state.analysis_complete = True
            
            return state
            
        except Exception as e:
            state.error_message = f"Error finalizing analysis: {str(e)}"
            return state
    
    async def evaluate_episode(self, episode_data: Dict) -> Dict[str, Any]:
        """Evaluate a single episode"""
        try:
            # Create episode object
            episode = KuhnPokerEpisode(**episode_data)
            
            # Create initial state
            initial_state = EvaluationState(episode=episode)
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            if final_state.error_message:
                return {"error": final_state.error_message}
            
            return {
                "episode_id": episode.episode_id,
                "scores": final_state.evaluation_scores,
                "predictions": final_state.llm_predictions,
                "success": final_state.analysis_complete
            }
            
        except Exception as e:
            return {"error": f"Failed to evaluate episode: {str(e)}"}

# ===== BATCH EVALUATION AND RESULTS AGGREGATION =====

class KuhnPokerBenchmark:
    """Main benchmark class for batch evaluation"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.evaluator = KuhnPokerEvaluator(self.llm)
        self.results = []
    
    async def evaluate_dataset(self, episodes: List[Dict]) -> pd.DataFrame:
        """Evaluate entire dataset of episodes"""
        print(f"Starting evaluation of {len(episodes)} episodes...")
        
        # Process episodes in batches to avoid rate limits
        batch_size = 5
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i+batch_size]
            batch_tasks = [self.evaluator.evaluate_episode(episode) for episode in batch]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Error in batch: {result}")
                else:
                    self.results.append(result)
            
            print(f"Completed {min(i+batch_size, len(episodes))}/{len(episodes)} episodes")
        
        # Convert to DataFrame for analysis
        return self._create_results_dataframe()
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create results DataFrame"""
        rows = []
        
        for result in self.results:
            if "error" in result:
                continue
                
            scores = result.get("scores", {})
            rows.append({
                "episode_id": result["episode_id"],
                "action_accuracy": scores.get("action_accuracy", 0.0),
                "optimal_accuracy": scores.get("optimal_accuracy", 0.0),
                "average_confidence": scores.get("average_confidence", 0.0),
                "total_reward": scores.get("total_reward", 0.0),
                "episode_length": scores.get("episode_length", 0),
                "decisions_evaluated": scores.get("decisions_evaluated", 0)
            })
        
        return pd.DataFrame(rows)
    
    def save_results(self, df: pd.DataFrame, filename_prefix: str = "kuhn_poker_evaluation"):
        """Save evaluation results to files"""
        try:
            # Save CSV
            csv_filename = f"{filename_prefix}_results.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Results saved to {csv_filename}")
            
            # Generate and save report
            report = self.generate_report(df)
            report_filename = f"{filename_prefix}_report.md"
            with open(report_filename, "w") as f:
                f.write(report)
            print(f"Report saved to {report_filename}")
            
            # Save detailed JSON results
            json_filename = f"{filename_prefix}_detailed.json"
            detailed_results = {
                "summary_stats": df.describe().to_dict(),
                "episodes": self.results
            }
            with open(json_filename, "w") as f:
                json.dump(detailed_results, f, indent=2)
            print(f"Detailed results saved to {json_filename}")
            
            return {
                "csv_file": csv_filename,
                "report_file": report_filename,
                "json_file": json_filename
            }
            
        except Exception as e:
            print(f"Error saving results: {e}")
            return None
        """Generate evaluation report"""
        if df.empty:
            return "No successful evaluations to report."
        
        report = f"""
# Kuhn Poker LLM Evaluation Report

## Overall Performance
- Episodes Evaluated: {len(df)}
- Average Action Accuracy: {df['action_accuracy'].mean():.3f}
- Average Optimal Strategy Accuracy: {df['optimal_accuracy'].mean():.3f}
- Average Confidence: {df['average_confidence'].mean():.3f}
- Average Total Reward: {df['total_reward'].mean():.3f}

## Detailed Statistics
{df.describe().round(3).to_string()}

## Performance Distribution
- Perfect Action Accuracy: {(df['action_accuracy'] == 1.0).sum()} episodes ({(df['action_accuracy'] == 1.0).mean()*100:.1f}%)
- Perfect Optimal Accuracy: {(df['optimal_accuracy'] == 1.0).sum()} episodes ({(df['optimal_accuracy'] == 1.0).mean()*100:.1f}%)
- High Confidence (>0.8): {(df['average_confidence'] > 0.8).sum()} episodes ({(df['average_confidence'] > 0.8).mean()*100:.1f}%)
"""
        return report

# ===== EXAMPLE USAGE =====

async def main():
    """Example usage of the Kuhn Poker evaluation framework"""
    
    # Sample offline data (your format)
    sample_episodes = [
        {
            "episode_id": 0,
            "str_states": ["2 1 p", "2 1 pp"],
            "num_states": ["2", "1p"],
            "player_ids": [0, 1],
            "str_actions": ["Pass", "Pass"],
            "num_actions": [0, 0],
            "rewards": [1.0, -1.0]
        },
        {
            "episode_id": 1,
            "str_states": ["3 2", "3 2 b"],
            "num_states": ["3", "2b"],
            "player_ids": [0, 1],
            "str_actions": ["Bet", "Pass"],
            "num_actions": [1, 0],
            "rewards": [2.0, -2.0]
        }
    ]
    
    # Initialize benchmark
    benchmark = KuhnPokerBenchmark(model_name="gpt-4")
    
    # Run evaluation
    print("Starting Kuhn Poker evaluation...")
    results_df = await benchmark.evaluate_dataset(sample_episodes)
    
    # Save all results
    saved_files = benchmark.save_results(results_df, "kuhn_poker_evaluation")
    
    if saved_files:
        print("\nEvaluation completed successfully!")
        print(f"Files created:")
        for file_type, filename in saved_files.items():
            print(f"  - {file_type}: {filename}")
    else:
        print("Error saving results")
    
    return results_df

# Run the evaluation
if __name__ == "__main__":
    # Note: In a real environment, you would run:
    # results = asyncio.run(main())
    print("Kuhn Poker LangChain/LangGraph evaluation framework ready!")
    print("Use: results = asyncio.run(main()) to run the evaluation")