import os
import torch
from breakout import *
from model import AtariNet
from agent import Agent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device}")

env = DQNBreakout(device=device, render_mode='rgb_array')

model = AtariNet(nb_actions=4)

model.load_the_model()

agent = Agent(model=model,
              device=device,
              epsilon=1.0,
              nb_warmup=5000,
              nb_actions=4,
              learning_rate=0.0001,
              memory_capacity=1000000,
              batch_size=64)

agent.train(env=env, epochs=200000)
