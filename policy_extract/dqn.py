# train dqn agents on environment and save policy
# https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/breakthrough_dqn.py

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import json

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import exploitability, exploitability_descent
from open_spiel.python.policy import Policy

from .utils import eval_against_random_bots

FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("checkpoint_dir", "/tmp/dqn_test",
                    "Directory to save/load the agent models.")
flags.DEFINE_integer(
    "save_every", int(1e4),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.")
flags.DEFINE_integer(
    "eval_episodes", 100,
    "How many episodes to evaluate against random bots"
)

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e4),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")


def main(_):
    game = "breakthrough"
    num_players = 2

    env_configs = {"columns": 5, "rows": 5}
    # env_configs = {}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    
    exploitability_values = []

    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    with tf.Session() as sess:
        hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
        # pylint: disable=g-complex-comprehension
        agents = [
            dqn.DQN(
                session=sess,
                player_id=idx,
                state_representation_size=info_state_size,
                num_actions=num_actions,
                hidden_layers_sizes=hidden_layers_sizes,
                replay_buffer_capacity=FLAGS.replay_buffer_capacity,
                batch_size=FLAGS.batch_size) for idx in range(num_players)
        ]
        sess.run(tf.global_variables_initializer())

        episode_data = []

        for ep in range(FLAGS.num_train_episodes):
            # logging.info(f"Episodes: {ep}")
            episode_str_states = []
            episode_actions = []
            episode_rewards = []

            if (ep + 1) % FLAGS.eval_every == 0:
                r_mean = eval_against_random_bots(env, agents, random_agents, FLAGS.eval_episodes)
                logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)

            if (ep + 1) % FLAGS.save_every == 0:
                for agent in agents:
                    agent.save(FLAGS.checkpoint_dir)
                with open(f'./output/{game}_dqn.json', 'w') as f:
                    json.dump(episode_data, f)
                logging.info("Saved training data")

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                # record state
                str_state = str(env.get_state)
                episode_str_states.append(str_state)
                # record reward
                timestep_rewards = time_step.rewards
                if timestep_rewards:
                    episode_rewards.append(timestep_rewards)
                
                # select action
                if env.is_turn_based:
                    agent_output = agents[player_id].step(time_step)
                    # record action
                    episode_actions.append(env.get_state.action_to_string(agent_output.action))
                    action_list = [agent_output.action]
                else:
                    agents_output = [agent.step(time_step) for agent in agents]
                    episdode_actions = [env.get_state.action_to_string(agent_output.action) for agent_output in agents_output]
                    action_list = [agent_output.action for agent_output in agents_output]
                
                time_step = env.step(action_list)

            if time_step.last():
                # record final reward and state
                final_rewards = time_step.rewards
                episode_rewards.append(final_rewards)
                episode_str_states.append(str(env.get_state))

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)

            episode_data.append({
                "episode_id": int(ep),
                "str_states": episode_str_states,
                "actions": episode_actions,
                "rewards": episode_rewards
            })

        # Save as JSON
        with open(f'./output/{game}_dqn.json', 'w') as f:
            json.dump(episode_data, f)
        print("Saved training data")



if __name__ == "__main__":
    app.run(main)
