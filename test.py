import os
import torch
from breakout import *
from model import AtariNet
from agent import Agent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device}")

env = DQNBreakout(device=device, render_mode='human')

model = AtariNet(nb_actions=4)

model.load_the_model('models/model_iter_5000.pt')

agent = Agent(model=model,
              device=device,
              epsilon=0.001,
              nb_warmup=5000,
              nb_actions=4,
              learning_rate=0.00001,
              memory_capacity=1000000,
              batch_size=64)

agent.test(env=env)
