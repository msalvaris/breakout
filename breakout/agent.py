import os
import random
from collections import deque

import numpy as np
import torch
import torch.optim as optim
from skimage.color import rgb2gray
from skimage.transform import resize

from breakout import utils
from breakout.model import Network


def agent_default_config():
    C = utils.CfgNode()
    C.batch_size = 32
    C.train_start = 50000  # wait until this many episodes are recorded in memory
    C.memory_size = 400000
    C.lr = 0.00025
    C.load_model = False
    C.model_path = "models"
    C.logs_path = "logs"
    C.device = "cuda" if torch.cuda.is_available() else "cpu"
    C.action_size = 3
    return C


def to_one_hot(target, action_dim):
    batch_size = target.shape[0]
    onehot = torch.zeros(batch_size, action_dim)
    onehot[np.arange(batch_size), target] = 1
    return onehot


class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.render = False
        self.load_model = self.config.load_model
        # environment settings
        self.state_size = (4, 84, 84)
        self.action_size = self.config.action_size
        # parameters about epsilon
        self.epsilon = 1.0
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.0
        self.epsilon_decay_step = (
            self.epsilon_start - self.epsilon_end
        ) / self.exploration_steps
        # parameters about training
        self.batch_size = 32
        self.train_start = (
            self.config.train_start
        )  # wait until this many episodes are recorded in memory
        self.update_target_rate = 10000  # how often to update the model
        self.discount_factor = 0.99
        self.memory = deque(
            maxlen=self.config.memory_size
        )  # This is stored in RAM, can reduce memory usage by caching to disk
        self.no_op_steps = 30
        # build model
        self.model = Network().to(self.config.device)
        self.target_model = Network().to(self.config.device)
        self.update_target_model()

        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=self.config.lr, alpha=0.95, eps=0.01
        )

        self.avg_q_max, self.avg_loss = 0, 0
        self.unclipped_score, self.clipped_score = 0, 0

        self.q_log, self.loss_log = [], []
        self.unclipped_log, self.clipped_log = [], []

        self.device = self.config.device

        if self.load_model:
            self.model.load_weights(self.config.model_path)

    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def compute_loss(self, history, action, target):

        py_x = self.model(history)

        a_one_hot = to_one_hot(action, self.action_size).to(self.device)
        q_value = torch.sum(py_x * a_one_hot, dim=1)
        error = torch.abs(target - q_value)

        quadratic_part = torch.clamp(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = torch.mean(0.5 * (quadratic_part**2) + linear_part)

        return loss

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = torch.tensor(history / 255.0, dtype=torch.float).to(self.device)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(history).detach().cpu().numpy()
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = []
        next_history = []
        target = np.zeros(self.batch_size)
        action, reward, dead = [], [], []

        # <0, 1, 2, 3, 4  >
        # <s, a, r, s' dead>
        for i in range(self.batch_size):
            history.append(torch.tensor(mini_batch[i][0] / 255, dtype=torch.float))
            next_history.append(torch.tensor(mini_batch[i][3] / 255, dtype=torch.float))
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        history = torch.cat(history, dim=0).to(self.device)
        next_history = torch.cat(next_history, dim=0).to(self.device)

        target_value = self.target_model(next_history).detach().cpu().numpy()

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * np.max(target_value[i])

        target = torch.tensor(target, dtype=torch.float64).to(self.device)
        action = np.array(action)
        loss = self.compute_loss(history, action, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.avg_loss += loss.item()

    def save_model(self):
        torch.save(
            self.model.state_dict(), os.path.join(self.config.model_path, "model.pt")
        )

    def log_and_reset(self):
        self.q_log.append(self.avg_q_max)
        self.loss_log.append(self.avg_loss)
        self.unclipped_log.append(self.unclipped_score)
        self.clipped_log.append(self.clipped_score)

        np.save(
            os.path.join(self.config.logs_path, "q_log.npy"),
            np.array(self.q_log, dtype=np.float64),
        )
        np.save(
            os.path.join(self.config.logs_path, "loss_log.npy"),
            np.array(self.loss_log, dtype=np.float64),
        )
        np.save(
            os.path.join(self.config.logs_path, "unclipped_log.npy"),
            np.array(self.unclipped_log, dtype=np.float64),
        )
        np.save(
            os.path.join(self.config.logs_path, "clipped_log.npy"),
            np.array(self.clipped_log, dtype=np.float64),
        )

        self.avg_q_max, self.avg_loss = 0, 0
        self.unclipped_score, self.clipped_score = 0, 0


# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode="constant") * 255
    )
    return processed_observe
