import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple
from Monopoly_env import MonopolyEnv

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, n_actions)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


def obs_to_state(obs, player_index):
    pos = obs['position'][player_index].astype(np.float32)
    money = obs['money'][player_index].astype(np.float32)
    ownership = obs['ownership'].astype(np.float32)
    pos_norm = pos / 39.0
    money_norm = money / 2000.0
    ownership_onehot = (ownership == player_index).astype(np.float32)
    opp_onehot = ((ownership != -1).astype(np.float32)) - ownership_onehot
    state = np.concatenate(([pos_norm, money_norm], ownership_onehot, opp_onehot))
    return state


class Agent:
    def __init__(self, input_dim, n_actions, lr=1e-3, buffer_size=100000, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.policy = DQN(input_dim, n_actions).to(device)
        self.target = DQN(input_dim, n_actions).to(device)
        self.target.load_state_dict(self.policy.state_dict())

        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0
        self.eps = 1.0
        self.n_actions = n_actions

    def select_action(self, state, eps_end=0.05, eps_decay=1e-4):
        self.eps = max(eps_end, self.eps * (1 - eps_decay))
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q_vals = self.policy(state)
            return int(q_vals.argmax(dim=1).item())

    def optimize_model(self, batch_size=64, gamma=0.99):
        if len(self.buffer) < batch_size:
            return
        transitions = self.buffer.sample(batch_size)
        state_batch = torch.tensor(np.array(transitions.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(transitions.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        non_final_mask = torch.tensor([s is not None for s in transitions.next_state], device=self.device)
        non_final_next_states = torch.tensor(
            np.array([s for s in transitions.next_state if s is not None]),
            dtype=torch.float32,
            device=self.device
        ) if any(non_final_mask) else torch.empty((0, state_batch.shape[1]), device=self.device)

        q_values = self.policy(state_batch).gather(1, action_batch)
        next_q_values = torch.zeros((len(transitions.reward), 1), dtype=torch.float32, device=self.device)
        if non_final_next_states.size(0) > 0:
            next_q = self.target(non_final_next_states).max(1)[0].detach().unsqueeze(1)
            next_q_values[non_final_mask] = next_q

        expected_q = reward_batch + gamma * next_q_values
        loss = F.mse_loss(q_values, expected_q)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

    def sync_target(self):
        self.target.load_state_dict(self.policy.state_dict())


def create_agents(env, n_agents=2):
    obs, _ = env.reset()
    sample_state = obs_to_state(obs, 0)
    input_dim = sample_state.shape[0]
    n_actions = env.action_space.n
    return [Agent(input_dim, n_actions) for _ in range(n_agents)]
