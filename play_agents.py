#play_agents.py
import torch
import torch.optim as optim
import random
from collections import deque
from Monopoly_env import MonopolyEnv
from agent_training import DQN, obs_to_state
import keyboard

render_enabled = False

def toggle_render():
    global render_enabled
    render_enabled = not render_enabled
    print(f"[Render {'ON' if render_enabled else 'OFF'}]")

keyboard.add_hotkey("r", lambda: toggle_render())


# Hyperparameters
num_episodes = 10000
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995
lr = 1e-3
batch_size = 64
replay_size = 10000

# Render settings
render_interval = 50      
max_render_steps = 20     

# Initialize environment
env = MonopolyEnv()
obs, _ = env.reset()
input_dim = len(obs_to_state(obs, 0))
n_actions = env.action_space.n

# Initialize agents
agents = [DQN(input_dim, n_actions), DQN(input_dim, n_actions)]
optimizers = [optim.Adam(agent.parameters(), lr=lr) for agent in agents]

# Replay buffer
memory = deque(maxlen=replay_size)

def select_action(agent, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        return agent(state).argmax(dim=1).item()

def optimize(agent, optimizer):
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = agent(next_states).max(1)[0]
    target = rewards + gamma * next_q_values * (1 - dones)

    loss = torch.nn.functional.mse_loss(q_values, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
epsilon = epsilon_start
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        current = env.current_player
        state = torch.tensor(obs_to_state(obs, current), dtype=torch.float32).unsqueeze(0)

        action = select_action(agents[current], state, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        
        money_values = next_obs['money']
        max_money = int(max(money_values))  
        
        if max_money >= 22000:
            truncated = True
            print(f"Truncated game: Player money = {money_values}, max = {max_money}")
        
        done = terminated or truncated

        next_state = torch.tensor(obs_to_state(next_obs, current), dtype=torch.float32).unsqueeze(0)
        memory.append((state, action, reward, next_state, done))

        optimize(agents[current], optimizers[current])
        obs = next_obs
        step_count += 1

        #Render if toggledrrr
        if render_enabled:
            env.render()

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    print(f"Episode {episode+1}/{num_episodes} finished | epsilon={epsilon:.3f}")

# Save trained policies
torch.save(agents[0].state_dict(), "policy_agent_0.pt")
torch.save(agents[1].state_dict(), "policy_agent_1.pt")
