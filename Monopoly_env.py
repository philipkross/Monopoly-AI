import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MonopolyEnv(gym.Env):
    def __init__(self, n_players=2, max_turns=10000):
        super(MonopolyEnv, self).__init__()

        self.n_players = n_players
        self.board_size = 40
        self.starting_money = 1500
        self.roll = 0
        self.dice = (0, 0)
        self.max_turns = max_turns
        self.turn_count = 0

        # Game statistics tracking
        self.games_played = 0
        self.player_wins = np.zeros(n_players, dtype=np.int32)
        self.game_winner = -1

        self.board = self._init_board()
        self.property_groups = self._init_property_groups()

        # Track recent rent payments
        self.recent_rents = [[] for _ in range(n_players)]  # List of recent rents for each player
        self.max_rent_history = 3  # Keep last 3 rent payments


        # Keep original state space structure for compatibility
        self.observation_space = spaces.Dict({
            "position": spaces.Box(0, self.board_size - 1, shape=(self.n_players,), dtype=np.int32),
            "money": spaces.Box(0, 20000, shape=(self.n_players,), dtype=np.int32),  # Increased max
            "ownership": spaces.Box(-1, self.n_players - 1, shape=(self.board_size,), dtype=np.int32),
            "houses": spaces.Box(0, 4, shape=(self.board_size,), dtype=np.int32),
            "hotels": spaces.Box(0, 1, shape=(self.board_size,), dtype=np.int32),
            "in_jail": spaces.MultiBinary(self.n_players)
        })

        # Expanded action space: 0: do nothing, 1: buy property, 2: chance, 3: community, 4: pay jail, 5: build house/hotel
        self.action_space = spaces.Discrete(6)
        self.reset()

    def _init_board(self):
        # Enhanced board with realistic prices and rent values - Traditional Monopoly order
        board = [
            {"name": "GO", "type": "corner", "group": None},
            {"name": "Kalmar Nation", "type": "property", "group": "brown", "price": 60, "rent": [2, 10, 30, 90, 160, 250], "house_cost": 50},
            {"name": "Community Chest", "type": "community", "group": None},
            {"name": "Gotland Nation", "type": "property", "group": "brown", "price": 60, "rent": [4, 20, 60, 180, 320, 450], "house_cost": 50},
            {"name": "Income Tax", "type": "tax", "group": None, "amount": 200},
            {"name": "Reading Railroad", "type": "railroad", "group": "railroad", "price": 200, "rent": [25, 50, 100, 200]},
            {"name": "Gottsunda", "type": "property", "group": "lightblue", "price": 100, "rent": [6, 30, 90, 270, 400, 550], "house_cost": 50},
            {"name": "Chance", "type": "chance", "group": None},
            {"name": "Gottgottgrillen", "type": "property", "group": "lightblue", "price": 100, "rent": [6, 30, 90, 270, 400, 550], "house_cost": 50},
            {"name": "Luthagen", "type": "property", "group": "lightblue", "price": 120, "rent": [8, 40, 100, 300, 450, 600], "house_cost": 50},
            {"name": "Jail", "type": "corner", "group": None},
            {"name": "Filler", "type": "property", "group": "pink", "price": 140, "rent": [10, 50, 150, 450, 625, 750], "house_cost": 100},
            {"name": "Electric Company", "type": "utility", "group": "utility", "price": 150},
            {"name": "Filler", "type": "property", "group": "pink", "price": 140, "rent": [10, 50, 150, 450, 625, 750], "house_cost": 100},
            {"name": "Filler", "type": "property", "group": "pink", "price": 160, "rent": [12, 60, 180, 500, 700, 900], "house_cost": 100},
            {"name": "Pennsylvania Railroad", "type": "railroad", "group": "railroad", "price": 200, "rent": [25, 50, 100, 200]},
            {"name": "Filler", "type": "property", "group": "orange", "price": 180, "rent": [14, 70, 200, 550, 750, 950], "house_cost": 100},
            {"name": "Community Chest", "type": "community", "group": None},
            {"name": "Filler", "type": "property", "group": "orange", "price": 180, "rent": [14, 70, 200, 550, 750, 950], "house_cost": 100},
            {"name": "Filler", "type": "property", "group": "orange", "price": 200, "rent": [16, 80, 220, 600, 800, 1000], "house_cost": 100},
            {"name": "Free Parking", "type": "corner", "group": None},
            {"name": "Carolina Rediviva", "type": "property", "group": "red", "price": 220, "rent": [18, 90, 250, 700, 875, 1050], "house_cost": 150},
            {"name": "Chance", "type": "chance", "group": None},
            {"name": "Uppsala Slott", "type": "property", "group": "red", "price": 220, "rent": [18, 90, 250, 700, 875, 1050], "house_cost": 150},
            {"name": "Uppsala Domkyrka", "type": "property", "group": "red", "price": 240, "rent": [20, 100, 300, 750, 925, 1100], "house_cost": 150},
            {"name": "Centrum", "type": "railroad", "group": "railroad", "price": 200, "rent": [25, 50, 100, 200]},
            {"name": "Filler", "type": "property", "group": "yellow", "price": 260, "rent": [22, 110, 330, 800, 975, 1150], "house_cost": 150},
            {"name": "Filler", "type": "property", "group": "yellow", "price": 260, "rent": [22, 110, 330, 800, 975, 1150], "house_cost": 150},
            {"name": "Water Works", "type": "utility", "group": "utility", "price": 150},
            {"name": "Filler", "type": "property", "group": "yellow", "price": 280, "rent": [24, 120, 360, 850, 1025, 1200], "house_cost": 150},
            {"name": "Go To Jail", "type": "corner", "group": None},
            {"name": "Fyrishov tentamenslokal", "type": "property", "group": "green", "price": 300, "rent": [26, 130, 390, 900, 1100, 1275], "house_cost": 200},
            {"name": "Danmarksgatan 30", "type": "property", "group": "green", "price": 300, "rent": [26, 130, 390, 900, 1100, 1275], "house_cost": 200},
            {"name": "Community Chest", "type": "community", "group": None},
            {"name": "Bergsbrunnagatan 15", "type": "property", "group": "green", "price": 320, "rent": [28, 150, 450, 1000, 1200, 1400], "house_cost": 200},
            {"name": "Uppsala CentralStation", "type": "railroad", "group": "railroad", "price": 200, "rent": [25, 50, 100, 200]},
            {"name": "Chance", "type": "chance", "group": None},
            {"name": "Stocken", "type": "property", "group": "darkblue", "price": 350, "rent": [35, 175, 500, 1100, 1300, 1500], "house_cost": 200},
            {"name": "Luxury Tax", "type": "tax", "group": None, "amount": 100},
            {"name": "Snerikes", "type": "property", "group": "darkblue", "price": 400, "rent": [50, 200, 600, 1400, 1700, 2000], "house_cost": 200},
        ]
        
        # Initialize additional fields for backward compatibility
        for i, space in enumerate(board):
            space["houses"] = 0
            space["hotel"] = 0
            space["mortgaged"] = False
            # Keep old price structure as fallback
            if space["type"] in ["property", "railroad", "utility"] and "price" not in space:
                space["price"] = 100 + i * 20
                space["house_cost"] = 50
        
        return board

    def _init_property_groups(self):
        """Initialize property groups for monopoly detection"""
        groups = {
            "brown": [1, 3],
            "lightblue": [6, 8, 9],
            "pink": [11, 13, 14],
            "orange": [16, 18, 19],
            "red": [21, 23, 24],
            "yellow": [26, 27, 29],
            "green": [31, 32, 34],
            "darkblue": [37, 39],
            "railroad": [5, 15, 25, 35],
            "utility": [12, 28]
        }
        return groups

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = np.zeros(self.n_players, dtype=np.int32)
        self.money = np.full(self.n_players, self.starting_money, dtype=np.int32)
        self.ownership = np.full(self.board_size, -1, dtype=np.int32)
        self.in_jail = np.zeros(self.n_players, dtype=np.int32)
        self.houses = np.zeros(self.board_size, dtype=np.int32)
        self.hotels = np.zeros(self.board_size, dtype=np.int32)
        self.current_player = 0
        self.turn_count = 0
        self.doubles_count = 0
        self.jail_turns = np.zeros(self.n_players, dtype=np.int32)
        self.game_winner = -1
             
        self.recent_rents = [[] for _ in range(self.n_players)]
        


        # Reset board state
        for space in self.board:
            space["houses"] = 0
            space["hotel"] = 0
            space["mortgaged"] = False
        
        return self._get_obs(), {}

    def step(self, action):
        player = self.current_player
        reward = 0
        terminated = False
        truncated = False

        # Clear penalties and bonuses from previous step
        self._monopoly_penalties = {}
        self._rent_bonuses = {}
        self._stalemate_penalties = {}  


        # Handle building action first (can be done before rolling dice)
        if action == 5:
            build_result = self._handle_building_action(player)
            reward += build_result
            # After building, continue with normal turn (roll dice and move)

        # Roll dice
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        self.dice = (d1, d2)
        self.roll = d1 + d2
        is_doubles = (d1 == d2)

        # Jail logic - enhanced but compatible
        if self.in_jail[player]:
            self.jail_turns[player] += 1
            
            if action == 4 and self.money[player] >= 50:  # Pay to get out
                self.money[player] -= 50
                self.in_jail[player] = 0
                self.jail_turns[player] = 0
                reward += 1
            elif is_doubles:  # Roll doubles to get out
                self.in_jail[player] = 0
                self.jail_turns[player] = 0
                reward += 2
                self.positions[player] = (self.positions[player] + self.roll) % self.board_size
                reward += self._resolve_space(player, action)
            elif self.jail_turns[player] >= 3:  # Forced to pay after 3 turns
                if self.money[player] >= 50:
                    self.money[player] -= 50
                    self.in_jail[player] = 0
                    self.jail_turns[player] = 0
                    self.positions[player] = (self.positions[player] + self.roll) % self.board_size
                    reward += self._resolve_space(player, action)
                else:
                    # Bankruptcy due to jail
                    terminated = True
                    reward -= 20
            
            self._next_player()
            
            # Apply monopoly penalties and rent bonuses to other players
            penalty_rewards = {p: self._monopoly_penalties.get(p, 0) + self._rent_bonuses.get(p, 0) 
                             for p in range(self.n_players)}
            return self._get_obs(), reward, terminated, truncated, {"penalty_rewards": penalty_rewards}

        # Normal movement with GO collection
        old_pos = self.positions[player]
        self.positions[player] = (self.positions[player] + self.roll) % self.board_size
        
        # Collect $200 for passing GO
        if self.positions[player] < old_pos:
            self.money[player] += 200
            reward += 2
        
        reward += self._resolve_space(player, action)
        
        # Handle doubles
        if is_doubles:
            self.doubles_count += 1
            if self.doubles_count >= 3:
                # Go to jail for rolling 3 doubles
                self.positions[player] = 10
                self.in_jail[player] = 1
                self.jail_turns[player] = 0
                self.doubles_count = 0
                reward -= 5
                self._next_player()
            # If rolled doubles but not going to jail, player gets another turn
        else:
            self.doubles_count = 0
            self._next_player()

        # Bankruptcy check with auto-mortgage
        if self.money[player] < 0:
            if not self._attempt_raise_money(player):
                terminated = True
                reward += -50
                self.game_winner = self._get_winner()

        # Check win condition
        active_players = np.sum(self.money >= 0)
        if active_players <= 1:
            terminated = True
            if self.money[player] > 0:
                reward += 50
                self.game_winner = player
            else:
                self.game_winner = self._get_winner()
        
        # Check for stalemate - if both players own at least one property in all sets
        # Check for stalemate - treat as draw with negative reward
        if self._is_stalemated():
            truncated = True
            reward += -2  # Penalty for reaching stalemate
            print("Game truncated: Stalemate detected - both players penalized")
            
            # Apply stalemate penalty to all players
            if not hasattr(self, '_stalemate_penalties'):
                self._stalemate_penalties = {}
            for player_id in range(self.n_players):
                self._stalemate_penalties[player_id] = -5
        
        # Update game stats when game ends
        if terminated:
            self.games_played += 1
            if self.game_winner >= 0:
                self.player_wins[self.game_winner] += 1

        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            truncated = True

        # Apply monopoly penalties and rent bonuses to other players
        penalty_rewards = {p: self._monopoly_penalties.get(p, 0) + self._rent_bonuses.get(p, 0) 
                         for p in range(self.n_players)}
        
        all_penalties = {}
        for p in range(self.n_players):
            penalty_sum = (self._monopoly_penalties.get(p, 0) + 
                        self._rent_bonuses.get(p, 0) + 
                        self._stalemate_penalties.get(p, 0))
            all_penalties[p] = penalty_sum

        
        return self._get_obs(), reward, terminated, truncated, {"penalty_rewards": all_penalties}

    def _handle_building_action(self, player):
        """Handle strategic building action - AI chooses which property to develop"""
        # Find all monopolies owned by player that can be developed
        developable_properties = self._get_developable_properties(player)
        
        if not developable_properties:
            return 0  # Small penalty for invalid action
        
        # Choose the best property to develop (simple heuristic: highest rent potential)
        best_property = self._choose_best_development_property(player, developable_properties)
        
        if best_property is None:
            return 0
        
        return self._build_on_property(player, best_property)

    def _get_developable_properties(self, player):
        """Get list of properties that can be developed (have monopoly and can build)"""
        developable = []
        
        # Check each property group for monopolies
        for group_name, property_indices in self.property_groups.items():
            if group_name in ["railroad", "utility"]:  # Can't build on these
                continue
            
            # Check if player has monopoly
            if not self._has_monopoly(player, group_name):
                continue
            
            # Check each property in the monopoly
            for prop_idx in property_indices:
                space = self.board[prop_idx]
                if self.ownership[prop_idx] != player:
                    continue
                
                # Can build if not mortgaged and less than hotel
                if (not space.get("mortgaged", False) and 
                    space.get("houses", 0) < 5 and  # 5 = hotel
                    self.money[player] >= space.get("house_cost", 50)):
                    
                    # Check even building rule (can't build if more than 1 house difference)
                    if self._can_build_evenly(player, group_name, prop_idx):
                        developable.append(prop_idx)
        
        return developable

    def _can_build_evenly(self, player, group_name, target_property):
        """Check if building on target property violates even building rule"""
        group_properties = self.property_groups[group_name]
        min_houses = float('inf')
        
        # Find minimum houses in the group
        for prop_idx in group_properties:
            if self.ownership[prop_idx] == player:
                houses = self.board[prop_idx].get("houses", 0)
                min_houses = min(min_houses, houses)
        
        # Can build if target property has minimum houses (even building)
        target_houses = self.board[target_property].get("houses", 0)
        return target_houses <= min_houses

    def _choose_best_development_property(self, player, developable_properties):
        """Choose the best property to develop using strategic heuristics"""
        if not developable_properties:
            return None
        
        best_property = None
        best_score = -1
        
        for prop_idx in developable_properties:
            space = self.board[prop_idx]
            current_houses = space.get("houses", 0)
            
            # Calculate score based on rent increase and strategic value
            score = 0
            
            # Rent increase value
            if "rent" in space and current_houses < len(space["rent"]) - 1:
                current_rent = space["rent"][current_houses]
                next_rent = space["rent"][current_houses + 1]
                rent_increase = next_rent - current_rent
                score += rent_increase * 0.1  # Weight rent increase
            
            # Prioritize completing hotel conversion (houses 4->5)
            if current_houses == 4:
                score += 200  # High priority for hotel
            
            # Prioritize higher-value properties
            score += space.get("price", 0) * 0.05
            
            # Consider landing frequency (properties more likely to be landed on)
            landing_frequency = self._get_landing_frequency(prop_idx)
            score += landing_frequency * 50
            
            if score > best_score:
                best_score = score
                best_property = prop_idx
        
        return best_property

    def _get_landing_frequency(self, position):
        """Get estimated landing frequency for a position (simplified heuristic)"""
        # Simple heuristic based on distance from jail and common positions
        distance_from_jail = min(abs(position - 10), 40 - abs(position - 10))
        
        # Properties 6-8 moves from jail are landed on most frequently
        optimal_distance = 7
        frequency = 1.0 - abs(distance_from_jail - optimal_distance) * 0.05
        
        # Boost for orange properties (frequently landed on after jail)
        if position in [16, 18, 19]:  # Orange properties
            frequency += 0.3
        
        return max(0.1, frequency)

    def _build_on_property(self, player, property_index):
        """Build house or hotel on specific property"""
        space = self.board[property_index]
        current_houses = space.get("houses", 0)
        house_cost = space.get("house_cost", 50)
        
        # Check if can afford
        if self.money[player] < house_cost:
            return 0  # Penalty for trying to build without money(outdated)
        
        # Build house or convert to hotel
        if current_houses < 4:
            # Build house
            self.money[player] -= house_cost
            space["houses"] += 1
            self.houses[property_index] = space["houses"]
            
            reward = 5 + current_houses  # Increasing reward for more houses
            print(f"Player {player + 1} built house on {space['name']} (now {space['houses']} houses)")
            
        elif current_houses == 4:
            # Convert to hotel
            self.money[player] -= house_cost
            space["houses"] = 5  # Internal representation: 5 = hotel
            space["hotel"] = 1
            self.houses[property_index] = 0  # Reset houses array
            self.hotels[property_index] = 1  # Set hotel
            
            reward = 20  # High reward for hotel
            print(f"Player {player + 1} built hotel on {space['name']}")
            
        else:
            # Already has hotel
            return 0
        
        return reward


    def _add_rent_to_history(self, player, rent_amount, property_name):
        """Add rent payment to player's history"""
        rent_entry = {
            'amount': rent_amount,
            'property': property_name,
            'turn': self.turn_count
        }
        
        # Add to front of list
        self.recent_rents[player].insert(0, rent_entry)
        
        # Keep only the most recent entries
        if len(self.recent_rents[player]) > self.max_rent_history:
            self.recent_rents[player] = self.recent_rents[player][:self.max_rent_history]


    def _attempt_raise_money(self, player):
        """Attempt to raise money through mortgaging properties"""
        debt = abs(self.money[player])
        raised = 0
        
        # Mortgage unmortgaged properties owned by player
        for i, owner in enumerate(self.ownership):
            if owner == player and not self.board[i]["mortgaged"] and raised < debt:
                mortgage_value = self.board[i]["price"] // 2
                self.board[i]["mortgaged"] = True
                raised += mortgage_value
        
        self.money[player] += raised
        return self.money[player] >= 0

    def _next_player(self):
        self.current_player = (self.current_player + 1) % self.n_players
        # Reset doubles count when turn changes
        if self.current_player == 0:  # Full round completed
            self.doubles_count = 0

    def _resolve_space(self, player, action):
        pos = self.positions[player]
        space = self.board[pos]
        reward = 0

        # Property purchase (action 1)
        if action == 1 and space.get("price") and self.ownership[pos] == -1:
            if self.money[player] >= space["price"]:
                self.money[player] -= space["price"]
                self.ownership[pos] = player
                reward += 5  # Base reward for property acquisition
                
                # Check if this purchase completes a monopoly (set)
                property_group = space.get("group")
                if property_group and self._has_monopoly(player, property_group):
                    reward += 10  # Bonus for completing a set
                    print(f"Player {player + 1} completed {property_group} monopoly!")
                    
                    # Give negative reward to other players
                    for other_player in range(self.n_players):
                        if other_player != player:
                            if not hasattr(self, '_monopoly_penalties'):
                                self._monopoly_penalties = {}
                            self._monopoly_penalties[other_player] = -5

        elif self.ownership[pos] != -1 and self.ownership[pos] != player and not space.get("mortgaged", False):
            rent = self._calculate_rent(pos)
            owner = self.ownership[pos]
            
            # Transfer money
            self.money[player] -= rent
            self.money[owner] += rent
            
            # Track rent payment in history
            self._add_rent_to_history(player, rent, space['name'])
            
            # Calculate rewards based on rent amount
            if rent <= 50:  # Low rent
                rent_penalty = -1
                rent_bonus = 1
            elif rent <= 200:  # Medium rent
                rent_penalty = -3
                rent_bonus = 3
            elif rent <= 500:  # High rent
                rent_penalty = -5
                rent_bonus = 15
            else:  # Very high rent (hotels, developed monopolies)
                rent_penalty = -10
                rent_bonus = 20
            
            reward += rent_penalty  # Current player gets negative reward
            
            # Property owner gets positive reward (stored for multi-agent returns)
            if not hasattr(self, '_rent_bonuses'):
                self._rent_bonuses = {}
            self._rent_bonuses[owner] = rent_bonus
            if rent >= 250:
                print(f"Player {player + 1} paid ${rent} rent to Player {owner + 1} on {space['name']}")




        # Chance card (action 2)
        elif action == 2 and space["type"] == "chance":
            reward += self._draw_chance_card(player)

        # Community Chest (action 3)
        elif action == 3 and space["type"] == "community":
            reward += self._draw_community_card(player)

        # Tax spaces 
        elif space["type"] == "tax":
            tax_amount = space.get("amount", 200)
            self.money[player] -= tax_amount
            reward -= 2

        # Go To Jail
        elif space["name"] == "Go To Jail":
            self.positions[player] = 10
            self.in_jail[player] = 1
            self.jail_turns[player] = 0

        return reward

    def _calculate_rent(self, pos):
        
        space = self.board[pos]
        owner = self.ownership[pos]
        
        if space["type"] == "property":
            houses = space.get("houses", 0)
            
            # Handle hotel case (houses = 5 means hotel)
            if houses >= 5:
                rent_index = 5  # Hotel rent
            else:
                rent_index = houses
            
            # Use rent array if available, otherwise fall back to old calculation
            if "rent" in space and rent_index < len(space["rent"]):
                base_rent = space["rent"][rent_index]
            else:
                # Fallback to old calculation
                base_rent = space["price"] // 2
                house_bonus = min(houses, 4) * 25
                if houses >= 5:  # Hotel
                    house_bonus = 200
                base_rent += house_bonus
            
            # Double rent for monopoly without houses
            if houses == 0 and self._has_monopoly(owner, space["group"]):
                return base_rent * 2
            return base_rent
            
        elif space["type"] == "railroad":
            railroads_owned = sum(1 for i in self.property_groups["railroad"] if self.ownership[i] == owner)
            if "rent" in space and railroads_owned <= len(space["rent"]):
                return space["rent"][railroads_owned - 1]
            else:
                return 25 * (2 ** (railroads_owned - 1))  # Fallback: 25, 50, 100, 200
            
        elif space["type"] == "utility":
            utilities_owned = sum(1 for i in self.property_groups["utility"] if self.ownership[i] == owner)
            multiplier = 4 if utilities_owned == 1 else 10
            return self.roll * multiplier
            
        return 0

    def _has_monopoly(self, player, group):
        """Check if player has monopoly in a color group"""
        if group not in self.property_groups:
            return False
        group_properties = self.property_groups[group]
        return all(self.ownership[pos] == player for pos in group_properties)

    def _draw_chance_card(self, player):
        """Enhanced chance cards"""
        outcomes = [
            ("Advance to GO", lambda: self._move_to(player, 0)),
            ("Pay poor tax", -15),
            ("Bank pays dividend", 50),
            ("Go to Jail", lambda: self._go_to_jail(player)),
            ("Advance to Illinois Ave", lambda: self._move_to(player, 24)),
            ("Pay building repairs", lambda: -25 * sum(self.houses) - 100 * sum(self.hotels))
        ]
        
        description, effect = random.choice(outcomes)
        
        if callable(effect):
            delta = effect()
            if isinstance(delta, int):
                self.money[player] += delta
        else:
            delta = effect
            self.money[player] += delta
        
        return 1 if delta > 0 else -1

    def _draw_community_card(self, player):
        """Enhanced community chest cards"""
        outcomes = [
            ("Bank error in your favor", 200),
            ("Doctor's fee", -50),
            ("From sale of stock", 50),
            ("Holiday fund matures", 100),
            ("Income tax refund", 20),
            ("Life insurance matures", 100),
            ("Hospital fees", -100),
            ("School fees", -50),
            ("Go to Jail", lambda: self._go_to_jail(player))
        ]
        
        description, effect = random.choice(outcomes)
        
        if callable(effect):
            delta = effect()
            if isinstance(delta, int):
                self.money[player] += delta
        else:
            delta = effect
            self.money[player] += delta
        
        return 1 if delta > 0 else -1

    def _move_to(self, player, position):
        """Move player to specific position and collect GO if passed"""
        old_pos = self.positions[player]
        self.positions[player] = position
        # Collect GO money if passed
        if position == 0 or position < old_pos:
            return 200
        return 0

    def _go_to_jail(self, player):
        """Send player to jail"""
        self.positions[player] = 10
        self.in_jail[player] = 1
        self.jail_turns[player] = 0
        return -100

    def _get_obs(self):
       
        # Get basic info about buildable properties
        current_player = self.current_player
        developable_props = self._get_developable_properties(current_player)
        
        obs = {
            "position": self.positions.copy(),
            "money": self.money.copy(),
            "ownership": self.ownership.copy(),
            "houses": self.houses.copy(),
            "hotels": self.hotels.copy(),
            "in_jail": self.in_jail.copy(),
            "dice": np.array(self.dice, dtype=np.int32),
            "turn_count": self.turn_count,
            "developable_properties": len(developable_props),  # Number of properties that can be developed
            "monopolies_owned": sum(1 for group in self.property_groups.keys() 
                                  if group not in ["railroad", "utility"] and self._has_monopoly(current_player, group))
        }
        
        return obs

    def _get_winner(self):
        """Determine winner based on highest net worth among surviving players"""
        best_player = -1
        best_worth = -1
        
        for player_id in range(self.n_players):
            if self.money[player_id] >= 0:  # Player still alive
                worth = self._calculate_player_net_worth(player_id)
                if worth > best_worth:
                    best_worth = worth
                    best_player = player_id
        
        return best_player

    def _is_stalemated(self):
        """Check if the game is in stalemate - ALL sets have split ownership"""
        # Only check property sets that can form monopolies (not railroads/utilities/brown(almost unwinnable) for this check)
        monopolizable_groups = ["lightblue", "pink", "orange", "red", "yellow", "green", "darkblue"]
        
        # Check each group for split ownership
        for group_name in monopolizable_groups:
            if group_name not in self.property_groups:
                continue
                
            group_properties = self.property_groups[group_name]
            owners_in_group = set()
            
            # Check who owns properties in this group
            for prop_idx in group_properties:
                owner = self.ownership[prop_idx]
                if owner != -1:  # If property is owned
                    owners_in_group.add(owner)
            
            # If this group is not split (0 or 1 owner), no stalemate yet
            if len(owners_in_group) <= 1:
                return False
        
        # If we get here, ALL groups have split ownership
        print("stalemate detected: Most monopolizable property groups have split ownership")
        return True

    def render(self, mode='human'):
        # Initialize figure only once
        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, (self._ax, self._sidebar_ax) = plt.subplots(1, 2, figsize=(20, 12), 
                                                                  gridspec_kw={'width_ratios': [2.2, 1]})
            self._fig.suptitle('Monopoly AI Training board', fontsize=16, fontweight='bold')
        
        # Clear both axes
        self._ax.clear()
        self._sidebar_ax.clear()

        # Draw main board
        self._draw_board()
        
        # Draw sidebar
        self._draw_sidebar()
        
        # Update display
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.01)

    def _draw_board(self):
        
        ax = self._ax
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 11)
        ax.set_aspect('equal')
        ax.axis('off')

        # Property colors
        color_map = {
            "brown": "saddlebrown",
            "lightblue": "lightblue", 
            "pink": "hotpink",
            "orange": "orange",
            "red": "red",
            "yellow": "yellow",
            "green": "green",
            "darkblue": "darkblue",
            "railroad": "black",
            "utility": "lightgray"
        }

        # Board coordinates
        coords = (
        [(i, 0) for i in range(10, -1, -1)] +   # Bottom row (right → left) 0–10
        [(0, i) for i in range(1, 11)] +        # Left column (bottom → top) 11–20
        [(i, 10) for i in range(1, 11)] +       # Top row (left → right) 21–30
        [(10, i) for i in range(9, 0, -1)]      # Right column (top → bottom) 31–39
        )

        # Draw board spaces
        for i, (x, y) in enumerate(coords):
            space = self.board[i]
            facecolor = 'white'
            edgecolor = 'black'

            # Color by group
            if space.get("group") in color_map:
                facecolor = color_map[space["group"]]

            # Special spaces
            if space['type'] == 'corner':
                facecolor = 'lightgrey'
            elif space['type'] == 'chance':
                facecolor = 'orange'
            elif space['type'] == 'community':
                facecolor = 'cyan'
            elif space['type'] == 'tax':
                facecolor = 'red'

            # Show mortgaged properties
            if space.get("mortgaged", False):
                edgecolor = 'red'
                facecolor = 'lightcoral'

            # Draw property rectangle
            rect = patches.Rectangle((x, y), 1, 1, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
            ax.add_patch(rect)
            
            # Property name - truncate if too long
            name = space['name']
            if len(name) > 12:
                name = name[:10] + '...'
            
            ax.text(x + 0.5, y + 0.5, name, ha='center', va='center', 
                   fontsize=6, wrap=True, weight='bold' if self.ownership[i] != -1 else 'normal')

            # Show ownership with colored border
            if self.ownership[i] != -1:
                player_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
                owner_color = player_colors[self.ownership[i] % len(player_colors)]
                thick_rect = patches.Rectangle((x-0.05, y-0.05), 1.1, 1.1, 
                                             facecolor='none', edgecolor=owner_color, linewidth=4)
                ax.add_patch(thick_rect)

            # Enhanced house and hotel visualization
            houses = space.get("houses", 0)
            if houses > 0:
                if houses < 5:  # Houses (1-4)
                    for h in range(houses):
                        house_x = x + 0.1 + h * 0.2
                        house_y = y + 0.05
                        house_rect = patches.Rectangle((house_x, house_y), 0.15, 0.15, 
                                                     facecolor='darkgreen', edgecolor='white', linewidth=1)
                        ax.add_patch(house_rect)
                        # Add 'H' text
                        ax.text(house_x + 0.075, house_y + 0.075, 'H', ha='center', va='center', 
                               fontsize=6, color='white', weight='bold')
                else:  # Hotel (houses = 5)
                    hotel_rect = patches.Rectangle((x + 0.2, y + 0.05), 0.6, 0.25, 
                                                 facecolor='darkred', edgecolor='gold', linewidth=2)
                    ax.add_patch(hotel_rect)
                    # Add 'HOTEL' text
                    ax.text(x + 0.5, y + 0.175, 'HOTEL', ha='center', va='center', 
                           fontsize=7, color='white', weight='bold')

        # Draw players
        player_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for player_id, pos in enumerate(self.positions):
            if 0 <= pos < len(coords):
                x, y = coords[pos]
                # Offset players so they don't overlap
                offset_x = 0.2 + (player_id % 2) * 0.6
                offset_y = 0.2 + (player_id // 2) * 0.6
                color = player_colors[player_id % len(player_colors)]
                
                # Draw player token
                ax.plot(x + offset_x, y + offset_y, 'o', color=color, markersize=15, 
                       markeredgecolor='white', markeredgewidth=2)
                
                # Show player number
                ax.text(x + offset_x, y + offset_y, str(player_id + 1), 
                       ha='center', va='center', fontsize=8, color='white', weight='bold')

        # Enhanced game info with rent information
        title = f"Turn {self.turn_count} | Player {self.current_player + 1}'s turn | Roll: {self.roll}"
        if self.doubles_count > 0:
            title += f" | Doubles: {self.doubles_count}"
        
        # Add building info
        developable = len(self._get_developable_properties(self.current_player))
        if developable > 0:
            title += f" | Can Build on {developable} properties"
            
        ax.set_title(title, fontsize=12, weight='bold', pad=20)

    def _draw_sidebar(self):
        """Enhanced sidebar showing rent reward information"""
        sidebar = self._sidebar_ax
        sidebar.set_xlim(0, 1)
        sidebar.set_ylim(0, 1)
        sidebar.axis('off')
        
        # Player colors
        player_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        
        # Title and game statistics
        sidebar.text(0.5, 0.98, 'BALANCED RENT REWARDS', ha='center', va='top', 
                    fontsize=14, weight='bold', transform=sidebar.transAxes)
        
        # Overall game stats
        y_pos = 0.94
        sidebar.text(0.5, y_pos, f'Games Played: {self.games_played}', ha='center', va='top',
                    fontsize=11, weight='bold', transform=sidebar.transAxes)
        y_pos -= 0.04
        
        # Win rates
        if self.games_played > 0:
            for player_id in range(self.n_players):
                win_rate = (self.player_wins[player_id] / self.games_played) * 100
                color = player_colors[player_id % len(player_colors)]
                sidebar.text(0.5, y_pos, f'Player {player_id + 1}: {self.player_wins[player_id]} wins ({win_rate:.1f}%)', 
                           ha='center', va='top', fontsize=10, color=color, weight='bold', transform=sidebar.transAxes)
                y_pos -= 0.035
        
        # Separator line
        y_pos -= 0.02
        line = plt.Line2D([0.05, 0.95], [y_pos, y_pos], 
                        color='gray', linestyle='-', linewidth=2, transform=sidebar.transAxes)
        sidebar.add_line(line)
        y_pos -= 0.04
        
        
        
        # Current game title
        y_pos -= 0.02
        sidebar.text(0.5, y_pos, 'CURRENT GAME', ha='center', va='top', 
                    fontsize=12, weight='bold', transform=sidebar.transAxes)
        y_pos -= 0.05
        
        # Players side by side
        col_width = 0.48
        col_positions = [0.02, 0.52]
        
        for player_id in range(min(self.n_players, 2)):
            x_pos = col_positions[player_id]
            y_current = y_pos
            color = player_colors[player_id % len(player_colors)]
            
            # Player header with background
            header_rect = patches.Rectangle((x_pos, y_current - 0.02), col_width - 0.02, 0.04, 
                                          facecolor=color, alpha=0.2, transform=sidebar.transAxes)
            sidebar.add_patch(header_rect)
            
            sidebar.text(x_pos + 0.02, y_current, f"PLAYER {player_id + 1}", 
                        fontsize=11, weight='bold', color=color, transform=sidebar.transAxes)
            y_current -= 0.045
            
            # Player money
            sidebar.text(x_pos + 0.02, y_current, f"${self.money[player_id]:,}", 
                        fontsize=10, weight='bold', transform=sidebar.transAxes)
            y_current -= 0.035
            
            # Rent income potential
            rent_potential = self._calculate_rent_income_potential(player_id)
            if rent_potential > 0:
                sidebar.text(x_pos + 0.02, y_current, f"Rent Income: ${rent_potential:,}/turn", 
                            fontsize=9, weight='bold', color='darkgreen', transform=sidebar.transAxes)
                y_current -= 0.03
            
            # Building potential
            developable = len(self._get_developable_properties(player_id))
            monopolies = sum(1 for group in self.property_groups.keys() 
                           if group not in ["railroad", "utility"] and self._has_monopoly(player_id, group))
            
            sidebar.text(x_pos + 0.02, y_current, f"Monopolies: {monopolies}", 
                        fontsize=9, weight='bold', color='darkgreen', transform=sidebar.transAxes)
            y_current -= 0.03
            
            if developable > 0:
                sidebar.text(x_pos + 0.02, y_current, f"Can Build: {developable} props", 
                           fontsize=9, weight='bold', color='blue', transform=sidebar.transAxes)
                y_current -= 0.03
            
            # Player status
            status_items = []
            if self.in_jail[player_id]:
                status_items.append(("Jail", f"{self.jail_turns[player_id]} turns", 'red'))
            
            if player_id == self.current_player:
                status_items.append(("Turn", "Active", 'green'))
            
            for status_type, status_value, status_color in status_items:
                sidebar.text(x_pos + 0.02, y_current, f"{status_type}: {status_value}", 
                           fontsize=9, color=status_color, weight='bold' if status_type == 'Turn' else 'normal',
                           transform=sidebar.transAxes)
                y_current -= 0.03
            
            # Net worth
            net_worth = self._calculate_player_net_worth(player_id)
            sidebar.text(x_pos + 0.02, y_current, f"Worth: ${net_worth:,}", 
                        fontsize=9, weight='bold', color=color, transform=sidebar.transAxes)
            y_current -= 0.04
            
            # Most dangerous properties (highest rent)
            dangerous_props = self._get_highest_rent_properties(player_id)
            if dangerous_props:
                sidebar.text(x_pos + 0.02, y_current, "High Rent Properties:", 
                            fontsize=9, weight='bold', color='darkred', transform=sidebar.transAxes)
                y_current -= 0.025
                
                for prop_name, rent in dangerous_props[:2]:  # Show top 2
                    sidebar.text(x_pos + 0.04, y_current, f"{prop_name}: ${rent}", 
                               fontsize=8, color='darkred', transform=sidebar.transAxes)
                    y_current -= 0.02
            

            # Recent rent payments
            y_current -= 0.02
            sidebar.text(x_pos + 0.02, y_current, "Recent Rent Paid:", 
                        fontsize=9, weight='bold', color='darkred', transform=sidebar.transAxes)
            y_current -= 0.025
            
            if self.recent_rents[player_id]:
                for rent_entry in self.recent_rents[player_id]:
                    rent_text = f"${rent_entry['amount']} - {rent_entry['property'][:12]}"
                    if len(rent_entry['property']) > 12:
                        rent_text = f"${rent_entry['amount']} - {rent_entry['property'][:9]}..."
                    
                    sidebar.text(x_pos + 0.04, y_current, rent_text, 
                               fontsize=8, color='darkred', transform=sidebar.transAxes)
                    y_current -= 0.02
            else:
                sidebar.text(x_pos + 0.04, y_current, "No recent payments", 
                           fontsize=8, color='gray', style='italic', transform=sidebar.transAxes)


    def _calculate_rent_income_potential(self, player_id):
        """Calculate potential rent income from all properties owned"""
        total_potential = 0
        for prop_idx in range(self.board_size):
            if self.ownership[prop_idx] == player_id:
                # Estimate average rent (use current rent or basic calculation)
                rent = self._calculate_rent_for_property(prop_idx)
                total_potential += rent
        return total_potential

    def _calculate_rent_for_property(self, prop_idx):
        """Calculate rent for a specific property without dice dependency"""
        space = self.board[prop_idx]
        owner = self.ownership[prop_idx]
        
        if space["type"] == "property":
            houses = space.get("houses", 0)
            rent_index = min(houses, 5) if houses < 5 else 5
            
            if "rent" in space and rent_index < len(space["rent"]):
                base_rent = space["rent"][rent_index]
            else:
                base_rent = space["price"] // 2
                if houses >= 5:
                    base_rent += 200
                else:
                    base_rent += houses * 25
            
            # Double for monopoly without houses
            if houses == 0 and self._has_monopoly(owner, space["group"]):
                return base_rent * 2
            return base_rent
            
        elif space["type"] == "railroad":
            railroads_owned = sum(1 for i in self.property_groups["railroad"] if self.ownership[i] == owner)
            return 25 * (2 ** (railroads_owned - 1))
            
        elif space["type"] == "utility":
            return 35  # Average utility rent
            
        return 0

    def _get_highest_rent_properties(self, player_id):
        """Get list of highest rent properties owned by player"""
        properties = []
        for prop_idx in range(self.board_size):
            if self.ownership[prop_idx] == player_id:
                space = self.board[prop_idx]
                rent = self._calculate_rent_for_property(prop_idx)
                if rent > 0:
                    name = space['name']
                    if len(name) > 15:
                        name = name[:12] + '...'
                    properties.append((name, rent))
        
        # Sort by rent descending
        properties.sort(key=lambda x: x[1], reverse=True)
        return properties

    def _get_player_properties(self, player_id):
        """Get list of property indices owned by player"""
        return [i for i, owner in enumerate(self.ownership) if owner == player_id]

    def _get_group_color(self, group_name):
        """Get display color for property group"""
        color_map = {
            "brown": "saddlebrown",
            "lightblue": "cyan",
            "pink": "magenta",
            "orange": "orange", 
            "red": "red",
            "yellow": "gold",
            "green": "green",
            "darkblue": "blue",
            "railroad": "black",
            "utility": "gray"
        }
        return color_map.get(group_name, "black")

    def _calculate_player_net_worth(self, player_id):
        """Calculate total net worth of player including buildings"""
        net_worth = self.money[player_id]
        
        for prop_idx in self._get_player_properties(player_id):
            space = self.board[prop_idx]
            # Add property value (mortgage value if mortgaged)
            prop_value = space.get("price", 0)
            if space.get("mortgaged", False):
                prop_value = prop_value // 2
            net_worth += prop_value
            
            # Add house/hotel value (sell back at half price)
            houses = space.get("houses", 0)
            house_cost = space.get("house_cost", 0)
            if houses >= 5:  # Hotel
                net_worth += house_cost * 5 // 2  # Hotel sells for half of 5 houses
            else:
                net_worth += houses * house_cost // 2  # Houses sell for half price
        
        return net_worth

# Example usage and comprehensive test with rent rewards
if __name__ == "__main__":
    env = MonopolyEnv(n_players=2)
    obs, info = env.reset()
   
    # Set up comprehensive test scenario for rent rewards
    print("\nSetting up rent reward test scenario...")
    
    # Give Player 1 brown monopoly with houses
    env.ownership[1] = 0  # Kalmar Nation
    env.ownership[3] = 0  # Gotland Nation (Player 1 has brown monopoly)
    env.board[1]["houses"] = 3  # 3 houses on Kalmar Nation
    env.board[3]["houses"] = 2  # 2 houses on Gotland Nation
    env.houses[1] = 3
    env.houses[3] = 2
    env.money[0] = 1000
    
    # Give Player 2 lightblue monopoly with hotel
    env.ownership[6] = 1  # Gottsunda
    env.ownership[8] = 1  # Gottgottgrillen  
    env.ownership[9] = 1  # Luthagen (Player 2 has lightblue monopoly)
    env.board[6]["houses"] = 5  # Hotel on Gottsunda
    env.board[6]["hotel"] = 1
    env.hotels[6] = 1
    env.money[1] = 1200
    
    print("Initial setup:")
    print(f"Player 1: ${env.money[0]}, owns brown monopoly with houses")
    print(f"Player 2: ${env.money[1]}, owns lightblue monopoly with hotel")
    
    # Render initial state
    env.render()
    input("Press Enter to test rent payment scenarios...")
    
    # Test 1: Player 1 lands on Player 2's hotel (direct rent test)
    print("\nTEST 1: Player 1 lands on Player 2's hotel...")
    env.positions[0] = 6  # Move Player 1 to Gottsunda (hotel)
    env.current_player = 0
    
    old_money_p1 = env.money[0]
    old_money_p2 = env.money[1]
    
    # Test rent directly without dice movement
    reward_p1 = env._resolve_space(0, 0)
    
    rent_paid = old_money_p1 - env.money[0]
    
    print(f"Rent paid: ${rent_paid} (should be $550)")
    print(f"Player 1 reward: {reward_p1} (should be -5 or -10)")
    print(f"Player 2 bonus from info: {getattr(env, '_rent_bonuses', {}).get(1, 0)} (should be 15 or 20)")
    
    env.render()
    input("Press Enter for next test...")
    
    # Test 2: Player 2 lands on Player 1's houses (direct rent test)
    print("\nTEST 2: Player 2 lands on Player 1's houses...")
    env.positions[1] = 1  # Move Player 2 to Kalmar Nation (3 houses)
    env.current_player = 1
    
    old_money_p1 = env.money[0]
    old_money_p2 = env.money[1]
    
    # Test rent directly without dice movement
    reward_p2 = env._resolve_space(1, 0)
    
    rent_paid = old_money_p2 - env.money[1]
    
    print(f"Rent paid: ${rent_paid} (should be $90)")
    print(f"Player 2 reward: {reward_p2} (should be -3)")
    print(f"Player 1 bonus from info: {getattr(env, '_rent_bonuses', {}).get(0, 0)} (should be 3)")
    
    env.render()
    
    print("\nBalanced Rent Reward System Test Complete!")
    print("Key features demonstrated:")
    print("✓ Rent payer gets negative reward (scaled by rent amount)")
    print("✓ Property owner gets positive reward (scaled by rent amount)")
    print("✓ Rewards are proportional to rent tier (low/medium/high/very high)")
    print("✓ Multi-agent reward distribution via info dict")
    print("✓ Visual feedback in sidebar showing rent tiers and income potential")
    
    # Keep window open
    input("\nPress Enter to exit...")
    plt.ioff()
    plt.close()