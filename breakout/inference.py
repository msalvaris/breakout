import os
import random
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from gym.wrappers.record_video import RecordVideo

from breakout import utils
from breakout.agent import DQNAgent, agent_default_config, pre_processing


def get_config():

    C = utils.CfgNode()
    C.video_path = "video"
    # system
    C.system = utils.CfgNode()
    C.system.seed = 3407

    # agent
    C.agent = agent_default_config()
    C.agent.device = "cuda" if torch.cuda.is_available() else "cpu"
    C.agent.load_model = True
    return C


# def main():
#     # Run the agent and record to video

#     action_map = {
#         0:1,
#         1:2,
#         2:3
#     }

#     cfg = get_config()
#     cfg.merge_from_args(sys.argv[1:])

#     print(f"Running model {cfg.agent.model_path}")
#     print(f"Saving video to {cfg.video_path}")
#     print(f"running on device {cfg.agent.device}")
    
#     env = gym.make('BreakoutDeterministic-v4', render_mode="rgb_array")
#     agent = DQNAgent(cfg.agent)

#     print("< Playing Atari Breakout >")
#     global_step = 0
#     os.makedirs(cfg.video_path, exist_ok=True)
    
#     env = RecordVideo(env, video_folder = cfg.video_path,  name_prefix="eval_", episode_trigger = lambda episode_number: True)

    
#     for episode_num in range(num_eval_episodes):
#         # env reset for a fresh start
#         observation, info = env.reset()

#         ###
#         # Start the recorder
#         # env.start_video_recorder()
#         episode_over = False
#         while not episode_over:
#             if cfg.agent.load_model:
#                 state = pre_processing(observation)
#                 agent.get_action(state)
#                 # change action to real_action
#                 real_action = action_map[action]

#             else:
#                 action = env.action_space.sample()  # random sample

#             observation, reward, terminated, truncated, info = env.step(action)

#             episode_over = terminated or truncated


    # for _ in range(1000):
    #     action = env.action_space.sample()  # agent policy that uses the observation and info
    #     observation, reward, terminated, info = env.step(action)

    #     if terminated:
    #         observation, info = env.reset()

    # ####
    # # Don't forget to close the video recorder before the env!
    # env.close_video_recorder()

    # # Close the environment
    # env.close()
    # show_video()