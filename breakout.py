import collections
import cv2
import gymnasium as gym
import numpy as np
from PIL import Image
import torch

class DQNBreakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', repeat=4, device='cpu'):
        env = gym.make("BreakoutNoFrameskip-v4", render_mode = render_mode)

        super(DQNBreakout, self).__init__(env)

        self.image_shape = (84,84)
        self.repeat = repeat
        self.lives = env.unwrapped.ale.lives()
        self.frame_buffer = []
        self.device = device

    def step(self, action):
        total_reward = 0
        done = False

        for i in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward

            current_lives = info['lives']

            if current_lives < self.lives:
                total_reward -= 1
                self.lives = current_lives

            if self.lives == 0:
                done = True

            self.frame_buffer.append(observation)

            if done:
                break

        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_obs(max_frame)
        max_frame = max_frame.to(self.device)

        total_reward = torch.tensor(total_reward, device=self.device).view(1, -1).float()

        done = torch.tensor(done, device=self.device).view(1, -1)

        return max_frame, total_reward, done, info

    def reset(self):
        self.frame_buffer = []

        obs, _ = self.env.reset()

        self.lives = self.env.unwrapped.ale.lives()

        obs = self.process_obs(obs)

        return obs

    def process_obs(self, obs):

        ing = Image.fromarray(obs)
        ing = ing.resize(self.image_shape)
        ing = ing.convert("L")
        ing = np.array(ing)
        ing = torch.from_numpy(ing)
        ing = ing.unsqueeze(0)
        ing = ing.unsqueeze(0)
        ing = ing / 255.0

        ing = ing.to(self.device)

        return ing