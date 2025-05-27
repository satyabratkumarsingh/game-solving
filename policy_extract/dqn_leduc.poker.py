

from absl import app
from absl import flags
from absl import logging
import numpy as np
import os
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import exploitability
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("checkpoint_dir", "./temp/dqn_test",
                    "Directory to save/load the agent models.")
flags.DEFINE_integer(
    "save_every", int(1e4),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.")

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")

def main(_):
  game = "leduc_poker"
  num_players = 2

  env_configs = {"columns": 5, "rows": 5}
  env = rl_environment.Environment(game)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  exploitability_values = []

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

    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.save_every == 0:
        for i, agent in enumerate(agents):
            q_file = f"q_network_pid1_ep{ep + 1}.pth"
            target_file = f"target_q_network_pid1_ep{ep + 1}.pth"
            #agent.save_q_network(q_file)
            #agent.save_target_q_network(target_file)
            print(f"Saved Q-network and target Q-network for episode {ep + 1}.")
            avg_policy = agent.()  # Assuming `agent` has a method to get its average policy
            current_exploitability = exploitability.exploitability(env.game, avg_policy)
            exploitability_values.append((ep + 1, current_exploitability))

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = agents[player_id].step(time_step)
          action_list = [agent_output.action]
        else:
          agents_output = [agent.step(time_step) for agent in agents]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)

    # Extract data for plotting
    episodes = [x[0] for x in exploitability_values]
    exploitability_scores = [x[1] for x in exploitability_values]

    # Plot the exploitability curve
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, exploitability_scores, marker='o', linestyle='-', color='b')
    plt.title("Exploitability over Episodes", fontsize=16)
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Exploitability", fontsize=14)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
  app.run(main)