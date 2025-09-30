import torch
import time
import numpy as np
from Monopoly_env import MonopolyEnv
from agent_training import DQN, obs_to_state, get_action_mask

def watch_trained_agents(model_files, num_games=1, step_delay=1.0, enable_render=True):
    """
    Watch trained agents play Monopoly
    
    Args:
        model_files: List of paths to saved model files
        num_games: Number of games to watch
        step_delay: Seconds to wait between moves (for readability)
        enable_render: Whether to show console output of game state
    """
    
    # Initialize environment
    env = MonopolyEnv()
    obs, _ = env.reset()
    
    input_dim = len(obs_to_state(obs, 0))
    n_actions = env.action_space.n
    
    # Load trained agents
    agents = []
    for i, model_file in enumerate(model_files):
        print(f"Loading agent {i} from {model_file}")
        agent = DQN(input_dim, n_actions)
        try:
            agent.load_state_dict(torch.load(model_file, map_location='cpu'))
            agent.eval() 
            agents.append(agent)
            print(f"✓ Agent {i} loaded successfully")
        except FileNotFoundError:
            print(f"✗ Model file {model_file} not found!")
            return
        except Exception as e:
            print(f"✗ Error loading {model_file}: {e}")
            return
    
    print(f"\n🎮 Starting {num_games} demonstration game(s)")
    print("=" * 50)
    
    for game in range(num_games):
        print(f"\n🎲 Game {game + 1}/{num_games}")
        obs, _ = env.reset()
        done = False
        step_count = 0
        
        while not done:
            current_player = env.current_player
            current_agent = agents[current_player]
            
            # Get current state
            state = torch.tensor(obs_to_state(obs, current_player), dtype=torch.float32).unsqueeze(0)
            
            # Get action mask
            action_mask = get_action_mask(obs, current_player, env)
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
            
            # Agent selects best action (no exploration) WITH MASK
            with torch.no_grad():
                q_values = current_agent(state, mask_tensor)
                action = q_values.argmax(dim=1).item()
            
            if enable_render:
                print(f"\n--- Step {step_count + 1} ---")
                print(f"Player {current_player + 1}'s turn")
                print(f"Current position: {obs['position'][current_player]} ({env.board[obs['position'][current_player]]['name']})")
                print(f"Money: ${obs['money'][current_player]}")
                print(f"Valid actions: {get_valid_action_names(action_mask)}")
                print(f"Action chosen: {get_action_name(action)}")
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            # Show result
            if enable_render:
                print(f"New position: {obs['position'][current_player]} ({env.board[obs['position'][current_player]]['name']})")
                print(f"Money after move: ${obs['money'][current_player]}")
                print(f"Reward: {reward}")
                
                # Render the board
                env.render()
                
                # Check for game end
                if done:
                    winner = env.game_winner if hasattr(env, 'game_winner') and env.game_winner >= 0 else "Unknown"
                    print(f"\n🏆 Game Over! Winner: Player {winner + 1 if isinstance(winner, int) else winner}")
                    print(f"Final money: {obs['money']}")
                    print(f"Game length: {step_count} steps")
                
                if step_delay > 0:
                    time.sleep(step_delay)
            
            # Safety limit
            if step_count > 1000:
                print("⚠️ Game truncated at 1000 steps")
                break
        
        print(f"\n✓ Game {game + 1} completed in {step_count} steps")
    
    print("\n🎬 Demonstration complete!")

def get_action_name(action):
    """Convert action number to readable name"""
    action_names = {
        0: "Do nothing/Roll dice",
        1: "Buy property",
        2: "Draw chance card",
        3: "Draw community chest",
        4: "Pay jail fee",
        5: "Build house/hotel"
    }
    return action_names.get(action, f"Unknown action {action}")

def get_valid_action_names(action_mask):
    """Get list of valid action names from mask"""
    valid_actions = np.where(action_mask > 0)[0]
    return [get_action_name(a) for a in valid_actions]

if __name__ == "__main__":
    model_files = ["policy_agent_0.pt", "policy_agent_1.pt"]
    
    print("🎮 Monopoly AI Agent Demo")
    print("Watching trained agents play...")
    
    # Watch the agents play
    watch_trained_agents(
        model_files=model_files,
        num_games=1,           
        step_delay=0.5,        
        enable_render=True    
    )