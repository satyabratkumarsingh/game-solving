# train qlearning agents on environment and save policy
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner, random_agent
import numpy as np
import matplotlib.pyplot as plt
import json


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    num_players = len(trained_agents)
    sum_episode_rewards = np.zeros(num_players)
    for player_pos in range(num_players):
        cur_agents = random_agents[:]
        cur_agents[player_pos] = trained_agents[player_pos]
        for _ in range(num_episodes):
            time_step = env.reset()
            episode_rewards = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if env.is_turn_based:
                    agent_output = cur_agents[player_id].step(
                        time_step, is_evaluation=True)
                    action_list = [agent_output.action]
                else:
                    agents_output = [
                        agent.step(time_step, is_evaluation=True) for agent in cur_agents
                    ]
                    action_list = [agent_output.action for agent_output in agents_output]
                time_step = env.step(action_list)
                episode_rewards += time_step.rewards[player_pos]
            sum_episode_rewards[player_pos] += episode_rewards
    return sum_episode_rewards / num_episodes

# Create the environment
game = "tic_tac_toe"
env = rl_environment.Environment(game)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]

# Create the agents
agents = [
    tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
    for idx in range(num_players)
]
random_agents = [
    random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
    for idx in range(num_players)
]

episode_data = []

evaluated_rewards = []

for cur_episode in range(100000):
    episode_str_states = []
    episode_actions = []
    episode_rewards = []

    if cur_episode % 1000 == 0:
        print(f"Episodes: {cur_episode}")
        evaluation_rewards = eval_against_random_bots(env, agents, random_agents, 100)
        print(f"Episode {cur_episode} - Evaluation rewards: {evaluation_rewards}")
        evaluated_rewards.append(evaluation_rewards)
    time_step = env.reset()
    while not time_step.last():
        # record state
        str_state = str(env.get_state)
        episode_str_states.append(str_state)
        # record reward
        timestep_rewards = time_step.rewards
        if timestep_rewards:
            episode_rewards.append(timestep_rewards)
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        # record action
        episode_actions.append(env.get_state.action_to_string(agent_output.action))
        time_step = env.step([agent_output.action])

    if time_step.last():
        # record final reward and state
        final_rewards = time_step.rewards
        episode_rewards.append(final_rewards)
        episode_str_states.append(str(env.get_state))
    # Episode is over, step all agents with final info state.
    for agent in agents:
        agent.step(time_step)

    episode_data.append({
        "episode_id": int(cur_episode),
        "str_states": episode_str_states,
        "actions": episode_actions,
        "rewards": episode_rewards
    })

with open(f'./output/{game}_qlearning.json', 'w') as f:
    json.dump(episode_data, f)
print("Saved training data")

# save evaluation rewards vs episodes as plot
evaluated_rewards = np.array(evaluated_rewards)

# Plot the evaluation rewards vs. episodes
plt.figure(figsize=(10, 6))

# Loop through each agent's rewards and plot
for i in range(num_players):
    plt.plot(np.arange(0, len(evaluated_rewards)) * 1000, evaluated_rewards[:, i], label=f"Agent {i + 1}")

plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title(f"{game} - Evaluation Rewards vs Episodes")
plt.legend()
plt.grid(True)
plt.savefig(f'./figures/evaluated_reward_{game}.png')

print("Plot saved!")
print("Done!")