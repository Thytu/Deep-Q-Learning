import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.utils import shuffle
from collections import namedtuple

class Network(nn.Module):
    """Agent network"""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.out = nn.Linear(50, out_size)

    def forward(self, t):
        if len(t.shape) == 3:
            t = t.unsqueeze(0)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.fc3(t)
        t = F.relu(t)
        return self.out(t)


class Memory():
    """Agent memory"""

    def __init__(self, size):
        self.size = size
        self.memory = []
        self.sequence = namedtuple("sequence", ["state", "action", "reward", "new_state"])

    def add(self, state, action, reward, new_state):
        """Add transition to memory"""

        if len(self.memory) == self.size:
            self.memory = self.memory[1:]
        self.memory.append(self.sequence(state, action, reward, new_state))

    def shuffle(self):
        self.memory = shuffle(self.memory)


class Agent():
    """RL agent"""

    def __init__(self, in_size, out_size, epsilone=0, min_eps=0, eps_decay=0.99, gamma=0.99, learning_rate=0.001, batch_size=32, memory_size=25_000, update_target_every=1_500):
        self.epsilone = epsilone
        self.MIN_EPS = min_eps
        self.EPS_DECAY = eps_decay
        
        self.BATCH_SIZE = batch_size
        self.network = Network(in_size, out_size)
        self.ACTION_SPACE = out_size
        self.target_network = Network(in_size, out_size)
        self.target_network.load_state_dict(self.network.state_dict())
        self._update_target_counter = 0
        self.update_target_every = update_target_every
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()
        self.memory = Memory(memory_size)
        self.GAMMA = gamma

    def pick_action(self, state):
        """Use epsilon to pick action"""

        if np.random.rand() < self.epsilone:
            action = np.random.randint(0, self.ACTION_SPACE)
        else:
            action = torch.argmax(self.network(state)).item()
        return action

    def update_epsilone(self):
        """Update epsilone according to it's decay and it's min value"""

        self.epsilone = max(self.epsilone * self.EPS_DECAY, self.MIN_EPS)

    def __calc_target(self, reward, next_state):
        """Calc expectation"""

        if next_state is None:
            return reward
        preds = self.target_network(torch.tensor(next_state).float())
        best_q = preds.max(dim=0)[0].item()
        return reward + self.GAMMA * best_q

    def __train_in_batch(self, states, actions, rewards, next_states):
        """train agent over one batch"""

        targets = [self.__calc_target(r, nxt) for r, nxt in zip(rewards, next_states)]
        if states.shape[0] != self.BATCH_SIZE:
            return
        self.optimizer.zero_grad()

        predictions = self.network(states)

        expectations = predictions.data.numpy().copy()
        expectations[range(self.BATCH_SIZE), actions.int()] = targets
        expectations = torch.tensor(expectations)

        loss_v = self.mse_loss(predictions, expectations)
        loss_v.backward()
        self.optimizer.step()

    """train the agent over the whole memory"""
    def train(self):
        states = torch.tensor([seq.state for seq in self.memory.memory]).float()
        actions =torch.tensor([seq.action for seq in self.memory.memory]).float()
        rewards = torch.tensor([seq.reward for seq in self.memory.memory]).float()
        next_states = [seq.new_state for seq in self.memory.memory]

        for index in range(0, len(self.memory.memory), self.BATCH_SIZE):
            self._update_target_counter += 1 * self.BATCH_SIZE
            self.__train_in_batch(states[index: index + self.BATCH_SIZE], actions[index: index + self.BATCH_SIZE], rewards[index: index + self.BATCH_SIZE], next_states[index: index + self.BATCH_SIZE])

        if self._update_target_counter >= self.update_target_every:
            self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path, map_location=torch.device('cpu')):
        self.network.load_state_dict(torch.load(path, map_location=map_location))
        self.target_network.load_state_dict(torch.load(path, map_location=map_location))
