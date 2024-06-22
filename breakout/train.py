import gymnasium as gym
from breakout.agent import DQNAgent
import numpy as np
import random
from breakout.agent import pre_processing, agent_default_config
from breakout import utils
import torch
import os
from datetime import datetime
import sys


def run_episode(agent, env, init_history, e, global_step, step, start_life, device):
    action_map = {
        0:1,
        1:2,
        2:3
    }
    done = False
    dead = False
    history = init_history

    while not done:
        if agent.render:
            env.render()
        global_step += 1
        step += 1

        # get action for the current history and go one step in environment
        action = agent.get_action(history)
        # change action to real_action
        real_action = action_map[action]

        observe, reward, done, truncate, info = env.step(real_action)

        # pre-process the observation --> history
        next_state = pre_processing(observe)
        next_state = np.reshape([next_state], (1, 1, 84, 84))
        next_history = np.append(next_state, history[:, :3, :, :], axis = 1)

        q_val = agent.model(torch.tensor(history / 255, dtype = torch.float).to(device))
        q_val = q_val.detach().cpu().numpy()[0]
        agent.avg_q_max += np.max(q_val)

        # if the agent missed ball, agent is dead --> episode is not over
        if start_life > info['lives']:
            dead = True
            start_life = info['lives']

        agent.unclipped_score += reward
        reward = np.clip(reward, -1., 1.)
        agent.clipped_score += reward

        # save the sample <s, a, r, s'> to the replay memory
        agent.replay_memory(history, action, reward, next_history, dead)

        # every some time interval, train model
        agent.train_replay()

        # update the target model with model
        if global_step % agent.update_target_rate == 0:
            agent.update_target_model()

        # if agent is dead, then reset the history
        if dead:
            dead = False
        else:
            history = next_history

        # if done, plot the score over episodes
        if done:
            print("episode:", e,
                "  score:", agent.unclipped_score,
                "  clipped score:", agent.clipped_score,
                "  memory length:", len(agent.memory),
                "\nepsilon:", agent.epsilon,
                "  global_step:", global_step,
                "  average_q:", agent.avg_q_max / float(step),
                "  average loss:", agent.avg_loss / float(step))
            # e += 1
            agent.log_and_reset()
    return global_step,step

def get_config():

    C = utils.CfgNode()

    # system
    C.system = utils.CfgNode()
    C.system.seed = 3407
    C.system.work_dir = "./out/breakout"
    C.system.experiment_id = ""

    # agent
    C.agent = agent_default_config()
    C.agent.device = "cuda" if torch.cuda.is_available() else "cpu"
    return C


def main():
    cfg = get_config()
    cfg.merge_from_args(sys.argv[1:])

    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if not cfg.system.experiment_id:
        experiment_id = f"experiment_breakout_{current_time}"
    else:
        experiment_id = cfg.system.experiment_id

    print(f"Starting training {experiment_id}")
    print(f"running on device {cfg.agent.device}")
    
    workdir = os.path.join(cfg.system.work_dir, experiment_id)
    os.makedirs(workdir, exist_ok=True) # Check we are able to save results to appropriate location

    cfg.agent.logs_path = os.path.join(workdir, cfg.agent.logs_path)
    cfg.agent.model_path = os.path.join(workdir, cfg.agent.model_path)
    os.makedirs(cfg.agent.logs_path, exist_ok=True)
    os.makedirs(cfg.agent.model_path, exist_ok=True)

    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(cfg.agent)

    print("< Playing Atari Breakout >")
    global_step = 0
    
    e = 1
    try:
        while True:

            # 1 episode = 5 lives
            step, _, start_life = 0, 0, 5
            observe = env.reset()

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal initialisation
            for _ in range(random.randint(1, agent.no_op_steps)):
                observe, _, _, _ , _= env.step(1)

            # At start of episode, there is no preceding frame
            # So just copy initial states to make history
            state = pre_processing(observe)
            init_history = np.stack((state, state, state, state))
            init_history = np.expand_dims(init_history, axis = 0)

            global_step, step = run_episode(agent, env, init_history, e, global_step, step, start_life, cfg.agent.device)
            e += 1

            if e % 1000 == 0:
                agent.save_model()
    except KeyboardInterrupt:
        print("Training interrupted - Saving model")
        agent.save_model()

