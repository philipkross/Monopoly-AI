#play_agents.py
import torch
import torch.optim as optim
import random
import numpy as np
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


def get_action_mask(obs, player_index, env):
    """
    Generate action mask for current player.
    Returns binary array: 1 = valid action, 0 = invalid action
    """
    player = player_index
    pos = obs['position'][player]
    space = env.board[pos]
    
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[0] = 1  # "do nothing" always valid
    
    # Action 1: Can buy property?
    if (space.get("price") and 
        obs['ownership'][pos] == -1 and 
        obs['money'][player] >= space["price"]):
        mask[1] = 1
    
    # Action 2: Can draw chance?
    if space["type"] == "chance":
        mask[2] = 1
    
    # Action 3: Can draw community chest?
    if space["type"] == "community":
        mask[3] = 1
    
    # Action 4: Can pay jail fine?
    if obs['in_jail'][player] and obs['money'][player] >= 50:
        mask[4] = 1
    
    # Action 5: Can build?
    if obs.get('developable_properties', 0) > 0:
        mask[5] = 1
    
    return mask


# Hyperparameters
num_episodes = 15000
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

# Replay buffer - now stores masks too
memory = deque(maxlen=replay_size)

def select_action(agent, state, action_mask, epsilon):
    """Select action using epsilon-greedy with masking."""
    # Get valid actions
    valid_actions = np.where(action_mask > 0)[0]
    
    if len(valid_actions) == 0:
        # Fallback: shouldn't happen, but default to action 0
        return 0
    
    # Epsilon-greedy: explore only among valid actions
    if random.random() < epsilon:
        return random.choice(valid_actions)
    
    # Exploit: choose best valid action
    with torch.no_grad():
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
        q_values = agent(state, mask_tensor)
        return int(q_values.argmax(dim=1).item())

def optimize(agent, optimizer):
    """Optimize with action masking."""
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)

    states = torch.cat(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)
    masks = torch.tensor(np.array(masks), dtype=torch.float32)
    next_masks = torch.tensor(np.array(next_masks), dtype=torch.float32)

    # Get Q-values with masking
    q_values = agent(states, masks).gather(1, actions.unsqueeze(1)).squeeze()
    
    # Get next Q-values with masking
    next_q_values = agent(next_states, next_masks).max(1)[0]
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
        
        # Get action mask
        action_mask = get_action_mask(obs, current, env)
        
        # Select action with mask
        action = select_action(agents[current], state, action_mask, epsilon)
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Check for runaway money
        money_values = next_obs['money']
        max_money = int(max(money_values))  
        
        if max_money >= 22000:
            truncated = True
            print(f"Truncated game: Player money = {money_values}, max = {max_money}")
        
        done = terminated or truncated

        # Get next state and mask
        next_state = torch.tensor(obs_to_state(next_obs, current), dtype=torch.float32).unsqueeze(0)
        next_action_mask = get_action_mask(next_obs, current, env) if not done else np.zeros(n_actions, dtype=np.float32)
        
        # Store transition with masks
        memory.append((state, action, reward, next_state, done, action_mask, next_action_mask))

        # Optimize
        optimize(agents[current], optimizers[current])
        
        obs = next_obs
        step_count += 1

        # Render if enabled
        if render_enabled:
            env.render()

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    print(f"Episode {episode+1}/{num_episodes} finished | epsilon={epsilon:.3f}")

# Save trained policies
torch.save(agents[0].state_dict(), "policy_agent_0.pt")
torch.save(agents[1].state_dict(), "policy_agent_1.pt")
print("Training complete! Models saved.")