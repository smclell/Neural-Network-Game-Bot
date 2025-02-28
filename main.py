import matplotlib.pyplot as plt
import os
from datetime import datetime
import random
import time
import torch
import copy
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils as torch_utils
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gymnasium as gym
from collections import deque
from PIL import Image
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Queue

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device}")

path = 'para-data-loader-testing'

########################################################################
run_num = 8 #update this every run
########################################################################

class DQNBreakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', repeat=4, device='cpu'):
        env = gym.make("BreakoutNoFrameskip-v4", render_mode = render_mode)

        super(DQNBreakout, self).__init__(env)

        #self.image_shape = (84,84)
        self.repeat = repeat
        self.lives = env.unwrapped.ale.lives()
        self.frame_buffer = []

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((84,84)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

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

        # ing = Image.fromarray(obs)
        # ing = ing.resize(self.image_shape)
        # ing = ing.convert("L")
        # ing = np.array(ing)
        # ing = torch.from_numpy(ing)
        # ing = ing.unsqueeze(0)
        # ing = ing.unsqueeze(0)
        # ing = ing / 255.0

        # ing = ing.to(self.device)

        ing = self.transform(obs).unsqueeze(0).to(self.device)

        return ing

class AtariNet(nn.Module):
    def __init__(self, nb_actions=4):

        super(AtariNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, nb_actions)


    def forward(self, x):
        x = torch.Tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def save_the_model(self, run_num, weights_filename=f'{path}/models/run-{run_num}/latest.pt'):
        if not os.path.exists(f'{path}/models/run-{run_num}'):
            os.makedirs(f"{path}/models/run-{run_num}")
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, run_num, weights_filename=f'{path}/models/run-{run_num}/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")

        except:
            print(f"No weights file available at {weights_filename}")    

class CustomDataset(Dataset):
    def __init__(self, device, num_samples=1000):
        self.device = device
        self.num_samples = num_samples
        self.env = DQNBreakout(device=device, render_mode='rgb_array')
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        state = self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()  # Random action for demonstration
            next_state, reward, done, _ = self.env.step(action)
            yield state, action, reward, done, next_state
            state = next_state

class ReplayMemory:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0
        self.device = device
        self.memory_max_report = 0

    def insert(self, transition):
        transition = [item.to(self.device) for item in transition]
        self.memory.append(transition)

    def sample(self, batch_size=32):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)

        batch = zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch]
    
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10
    
    def __len(self):
        return len(self.memory)

class LivePlot():
    
    def __init__(self):
        self.fig, self.ax = plt.subplots()

        self.data = None
        self.eps_data = None

        self.epochs = 0

    def update_plot(self, stats, run_num):
        self.data = stats['AvgReturns']
        self.eps_data = stats['EpsilonCheckpoint']

        self.epochs = len(self.data)
        
        self.ax.clear()
        self.ax.set_xlim(0, self.epochs)

        self.ax.plot(self.data, 'b-', label='Rewards')
        self.ax.plot(self.eps_data, 'r-', label='Epsilon')
        self.ax.set_xlabel("Episodes x 10")
        self.ax.set_ylabel("Rewards")
        self.ax.set_title("Rewards over Episodes")
        self.ax.legend(loc='upper left')

        if not os.path.exists(f'{path}/plots/run-{run_num}'):
            os.makedirs(f'{path}/plots/run-{run_num}')

        #current_date = datetime.now().strftime('%Y-%m-%d--%H%M-%S')

        self.fig.savefig(f'{path}/plots/run-{run_num}/plot_iter_{self.epochs * 10}.png')

class Agent:
    def __init__(self, model, device='cpu', epsilon=1.0, min_epsilon=0.1, nb_warmup=10000, nb_actions=None,
                 memory_capacity=10000, batch_size=32, learning_rate=0.00025):
        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2)
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99
        self.nb_actions = nb_actions

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1, keepdim=True)

    def train(self, states, actions, rewards, dones, next_states):
        # Convert batched data to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        next_states = torch.stack(next_states)

        # Get Q-values for current states
        q_values = self.model(states)

        # Get Q-values for next states using the target network
        next_q_values = self.target_model(next_states).detach()
        max_next_q_values = torch.max(next_q_values, dim=-1, keepdim=True)[0]

        # Compute target Q-values
        target_q_values = rewards + ~dones * self.gamma * max_next_q_values

        # Gather the Q-values for the actions taken
        q_values_for_actions = q_values.gather(1, actions)

        # Compute the loss using mean squared error
        loss = F.mse_loss(q_values_for_actions, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Calculate returns for the current batch
        batch_return = rewards.sum().item()

        return batch_return  # Return the total return for the batch
    
    def test(self, env):
        for epoch in range(1, 3):
            state = env.reset()

            done = False
            for _ in range(1000):
                time.sleep(0.01)
                action = self.get_action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break

num_epochs = 9000

def worker(agent, env, data_loader, epochs_per_process, process_index):
    start_epoch = process_index * epochs_per_process
    end_epoch = min((process_index + 1) * epochs_per_process, num_epochs)

    stats = {'Returns': [], 'AvgReturns': [], 'EpsilonCheckpoint': []}

    for epoch in range(start_epoch, end_epoch):
        epoch_return = 0  # Accumulate returns across batches

        for batch in data_loader:
            states, actions, rewards, dones, next_states = batch

            # Train the agent on the current batch and get the total return
            total_return = agent.train(states, actions, rewards, dones, next_states)

            # Accumulate returns
            epoch_return += total_return

        # Calculate average return for the epoch
        average_return = epoch_return / len(data_loader)

        # Update epsilon
        if agent.epsilon > agent.min_epsilon:
            agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
        stats['EpsilonCheckpoint'].append(agent.epsilon)

        # Update average returns for display
        stats['AvgReturns'].append(average_return)

        # Update process_stats with the current stats
        process_stats[process_index] = stats.copy()

    # Return the final stats
    return stats

       

########################################################################
#    UPDATE THIS TO RUN AS ONLY ONE ENVIRONMENT TO SEE HOW IT GOES     #
########################################################################

if __name__ == '__main__':
    # Define the number of worker processes
    num_processes = 4

    # Divide epochs among processes
    epochs_per_process = num_epochs // num_processes

    # Create a custom dataset
    dataset = CustomDataset(device=device, num_samples=1000)
    
    # Create a DataLoader for your dataset
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    progress_bars = []
    process_stats = [None] * num_processes

    # Spawn worker processes
    processes = []
    for i in range(num_processes):
        # Create DQN environment and agent
        env = DQNBreakout(device=device, render_mode='rgb_array')
        model = AtariNet(nb_actions=4).to(device)
        agent = Agent(model=model, device=device, epsilon=1.0, nb_warmup=5000, nb_actions=4, learning_rate=0.0001, memory_capacity=100000, batch_size=32)

        progress_bar = tqdm(total=epochs_per_process, desc=f"Process {i+1} Progress")
        progress_bars.append(progress_bar)

        process_stats.append({'Returns': [], 'AvgReturns': [], 'EpsilonCheckpoint': []})

        p = Process(target=worker, args=(agent, env, data_loader, epochs_per_process, i, process_stats))
        p.start()
        processes.append(p)

    # pid = 0
    # for p in processes:
    #     pid += 1
    #     if (p.is_alive()):
    #         print(f"Process {pid} is running")

    average_reward = 0
    # Initialize progress bar
    with tqdm(total=num_epochs, desc="Training Progress") as overall_pbar:
        # Count the number of updates received from each process
        updates_received = [0] * num_processes

        while any(p.is_alive() for p in processes):
            # Check for progress updates from worker processes
            for i, p in enumerate(processes):
                p.join(timeout=0.1)
                if not p.is_alive():
                    # Update progress bar based on the epoch and process index
                    progress_bars[i].update(epochs_per_process)
                    
                    # Get the stats returned by the process
                    stats = process_stats[i]
                    # Calculate the average return over the last few epochs
                    avg_return = np.mean(stats['AvgReturns'][-10:])  # Adjust window size as needed
                    # Update the progress bar with the average return as postfix
                    progress_bars[i].set_postfix(Average_Return=avg_return)

                    # Update the overall progress bar
                    overall_pbar.update(epochs_per_process)
                    sum_updates = sum(updates_received)
                    overall_pbar.set_postfix(Updates=sum_updates)

                    updates_received[i] += epochs_per_process

            # Sleep briefly to avoid excessive CPU usage
            time.sleep(0.1)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All processes finished.")

# 

# env = DQNBreakout(device=device, render_mode='rgb_array')

# model = AtariNet(nb_actions=4).to(device)

# model.load_the_model(run_num) #weights_filename='models\model_iter_5000.pt'

# agent = Agent(model=model,
#               device=device,
#               epsilon=1.0,
#               nb_warmup=5000,
#               nb_actions=4,
#               learning_rate=0.0001,
#               memory_capacity=100000,
#               batch_size=32)

# with tqdm(total=num_epochs, desc="Training Progress") as pbar:
#     agent.train(env=env, epochs=num_epochs, progress_bar=pbar)