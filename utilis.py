import numpy as np
import torch
from tqdm import tqdm
from typing import NamedTuple
from collections import deque
import random
import gymnasium as gym


def npTorchCGPU_manula_seed(device, myseed=1234):
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if device != "cpu":
        torch.cuda.manual_seed(myseed)

class Params(NamedTuple):
    input_dim: int
    hidden_dim: int
    in_planes: int
    planes: int
    trgQ_update_freq: int
    buffer_size: int
    buffer_min_size: int
    batch_size: int
    total_episodes: int
    learning_rate: float
    wd: float
    gamma: float  # Discounting rate
    seed: int
    action_size: int
    value: str
    device: str
    epsilon: float


class Buffer:
    def __init__(self, maxlen, batch_size):
        self.memo = deque([], maxlen=maxlen)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        self.memo.append([state, action, reward, next_state, done])

    def replay(self):
        experiences = random.sample(self.memo, min(len(self.memo), self.batch_size))
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        dones = torch.cat(dones)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memo)


class DataCollect:
    def __init__(self):
        self.hashmap = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}

    def push(self, state, action, next_state, reward, done):
        self.hashmap["states"].append(state)
        self.hashmap["actions"].append(action)
        self.hashmap["next_states"].append(next_state)
        self.hashmap["rewards"].append(reward)
        self.hashmap["dones"].append(done)

    def pop(self):
        states = torch.cat(self.hashmap["states"])
        actions = torch.cat(self.hashmap["actions"])
        rewards = torch.cat(self.hashmap["rewards"])
        next_states = torch.cat(self.hashmap["next_states"])
        dones = torch.cat(self.hashmap["dones"])

        return states, actions, rewards, next_states, dones


class RLTrainer:
    def __init__(self, env_obj):
        self.env_obj = env_obj

    def train(self, policy_mode, learner, **kwargs):
        if policy_mode == "on-policy":
            self._on_policy(learner)
        elif policy_mode == "off-policy":
            if "buffer" not in kwargs:
                raise ValueError("buffer is not provided in off-policy")
            self._off_policy(learner, kwargs["buffer"])
        else:
            raise ValueError("please provide policy_mode correctly")

    def _on_policy(self, learner):
        for episode in tqdm(range(self.env_obj.params.total_episodes), desc=f"Episodes"):
            state = self.env_obj.env_reset()
            done = False

            collator = DataCollect()
            while not done:
                self.env_obj.steps[episode] += 1

                action = learner.take_action(state)
                next_state, reward, done, info = self.env_obj.env_step(action.item())
                self.env_obj.episode_rewards[episode] += reward

                collator.push(state, action, next_state, reward, done)
                state = next_state

            states, actions, rewards, next_states, dones = collator.pop()
            learner.update(states, actions, rewards, next_states, dones, episode, episode % self.env_obj.params.trgQ_update_freq == 0)

    def _off_policy(self, learner, buffer):
        for episode in tqdm(range(self.env_obj.params.total_episodes), desc=f"Episodes"):
            state = self.env_obj.env_reset()
            done = False

            while not done:
                self.env_obj.steps[episode] += 1

                action = learner.take_action(state)
                next_state, reward, done, info = self.env_obj.env_step(action.item())
                self.env_obj.episode_rewards[episode] += reward

                buffer.push(state, action, reward, next_state, done)
                state = next_state

                if len(buffer) < self.env_obj.params.buffer_min_size:
                    continue
                states, actions, rewards, next_states, dones = buffer.replay()

                learner.update(states, actions, rewards, next_states, dones, episode, self.env_obj.steps[episode] % self.env_obj.params.trgQ_update_freq == 0)


class TaskEnv:
    def __init__(self, task, action_map, params):
        self.env = gym.make(**task)
        self.action_map = action_map
        self.params = params

        self.steps = np.zeros(params.total_episodes)
        self.episode_rewards = np.zeros_like(self.steps)

    def env_reset(self):
        state = torch.tensor(self.env.reset(seed=self.params.seed)[0]).to(self.params.device)
        state = state.unsqueeze(0)
        if len(self.env.observation_space.shape) > 1:
            state = state.transpose(1, 3).float()
        return state

    def env_step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(self.action_map[action])
        next_state = torch.tensor(next_state).unsqueeze(0)
        if len(self.env.observation_space.shape) > 1:
            next_state = next_state.transpose(1, 3).float()
        return next_state.to(self.params.device), torch.tensor([reward]).to(self.params.device), torch.tensor([terminated or truncated]).to(self.params.device), info
