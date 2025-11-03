

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import random


class ProceduralDungeonEnv(gym.Env):
    """
    Multi-agent procedural dungeon environment.

    Features:
    - Procedurally generated maps (different every episode)
    - Multiple agents must coordinate to defeat enemies
    - Treasures provide team rewards
    - Obstacles block movement
    - Partial observability (agents see limited range)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        num_agents: int = 2,
        grid_size: int = 10,
        num_enemies: int = 3,
        num_treasures: int = 2,
        num_obstacles: int = 8,
        difficulty: float = 0.5,
        vision_range: int = 3,
        max_steps: int = 200,
        seed: Optional[int] = None
    ):
        super().__init__()

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_enemies = num_enemies
        self.num_treasures = num_treasures
        self.num_obstacles = num_obstacles
        self.difficulty = difficulty  # 0.0 = easy, 1.0 = hard
        self.vision_range = vision_range
        self.max_steps = max_steps

        if seed is not None:
            self.seed(seed)

        # Entity IDs
        self.EMPTY = 0
        self.WALL = 1
        self.AGENT = 2
        self.ENEMY = 3
        self.TREASURE = 4
        self.OBSTACLE = 5

        # Action space: [up, down, left, right, attack, stay]
        self.action_space = spaces.Discrete(6)

        # Observation space: local grid around agent + agent stats
        obs_grid_size = (2 * vision_range + 1) ** 2
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0, high=5,
                shape=(obs_grid_size,),
                dtype=np.int32
            ),
            'agent_health': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'team_health': spaces.Box(low=0, high=100, shape=(num_agents,), dtype=np.float32),
            'position': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            'enemies_remaining': spaces.Box(low=0, high=num_enemies, shape=(1,), dtype=np.int32),
        })

        # State
        self.grid = None
        self.agent_positions = None
        self.agent_healths = None
        self.enemy_positions = None
        self.enemy_healths = None
        self.treasure_positions = None
        self.treasures_collected = 0
        self.current_step = 0

    def seed(self, seed: int):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)

    def _generate_procedural_map(self):
        """Generate a new random dungeon layout"""
        # Initialize empty grid with walls on borders
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

        # Place obstacles (scaled by difficulty)
        num_obs = int(self.num_obstacles * (1 + self.difficulty))
        for _ in range(num_obs):
            x, y = self._get_empty_position()
            self.grid[x, y] = self.OBSTACLE

        # Place agents
        self.agent_positions = []
        self.agent_healths = np.full(self.num_agents, 100.0, dtype=np.float32)
        for i in range(self.num_agents):
            x, y = self._get_empty_position()
            self.agent_positions.append([x, y])
            self.grid[x, y] = self.AGENT

        # Place enemies (scaled by difficulty)
        num_enemies = int(self.num_enemies * (1 + 0.5 * self.difficulty))
        self.enemy_positions = []
        self.enemy_healths = []
        for _ in range(num_enemies):
            x, y = self._get_empty_position()
            self.enemy_positions.append([x, y])
            enemy_health = 50 + 50 * self.difficulty  # Harder enemies have more health
            self.enemy_healths.append(enemy_health)
            self.grid[x, y] = self.ENEMY

        # Place treasures
        self.treasure_positions = []
        for _ in range(self.num_treasures):
            x, y = self._get_empty_position()
            self.treasure_positions.append([x, y])
            self.grid[x, y] = self.TREASURE

    def _get_empty_position(self) -> Tuple[int, int]:
        """Find a random empty position on the grid"""
        while True:
            x = np.random.randint(1, self.grid_size - 1)
            y = np.random.randint(1, self.grid_size - 1)
            if self.grid[x, y] == self.EMPTY:
                return x, y

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        if seed is not None:
            self.seed(seed)

        self._generate_procedural_map()
        self.current_step = 0
        self.treasures_collected = 0

        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def _get_observations(self) -> List[Dict]:
        """Get observations for all agents"""
        observations = []
        for i in range(self.num_agents):
            obs = self._get_agent_observation(i)
            observations.append(obs)
        return observations

    def _get_agent_observation(self, agent_id: int) -> Dict:
        """Get observation for a specific agent (partial observability)"""
        x, y = self.agent_positions[agent_id]

        # Extract local grid around agent
        local_grid = []
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    local_grid.append(self.grid[nx, ny])
                else:
                    local_grid.append(self.WALL)  # Out of bounds = wall

        return {
            'grid': np.array(local_grid, dtype=np.int32),
            'agent_health': np.array([self.agent_healths[agent_id]], dtype=np.float32),
            'team_health': self.agent_healths.copy(),
            'position': np.array(self.agent_positions[agent_id], dtype=np.int32),
            'enemies_remaining': np.array([len(self.enemy_positions)], dtype=np.int32),
        }

    def step(self, actions: List[int]) -> Tuple[List[Dict], List[float], bool, bool, Dict]:
        """
        Execute one step with actions from all agents.

        Args:
            actions: List of actions (one per agent)

        Returns:
            observations, rewards, terminated, truncated, info
        """
        self.current_step += 1

        # Store old positions for collision detection
        old_positions = [pos.copy() for pos in self.agent_positions]

        # Execute agent actions
        individual_rewards = []
        for i, action in enumerate(actions):
            reward = self._execute_agent_action(i, action, old_positions)
            individual_rewards.append(reward)

        # Enemy AI (simple: move towards nearest agent and attack)
        self._update_enemies()

        # Calculate rewards
        rewards = self._calculate_rewards(individual_rewards)

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps

        observations = self._get_observations()
        info = self._get_info()

        return observations, rewards, terminated, truncated, info

    def _execute_agent_action(self, agent_id: int, action: int, old_positions: List) -> float:
        """Execute single agent action and return individual reward"""
        reward = -0.01  # Small step penalty
        x, y = self.agent_positions[agent_id]

        # Clear old position
        self.grid[x, y] = self.EMPTY

        # Action: 0=up, 1=down, 2=left, 3=right, 4=attack, 5=stay
        new_x, new_y = x, y

        if action == 0:  # Up
            new_x = x - 1
        elif action == 1:  # Down
            new_x = x + 1
        elif action == 2:  # Left
            new_y = y - 1
        elif action == 3:  # Right
            new_y = y + 1
        elif action == 4:  # Attack
            reward += self._attack_adjacent_enemy(agent_id)
        elif action == 5:  # Stay
            pass

        # Check if new position is valid
        if action < 4:  # Movement action
            if self._is_valid_position(new_x, new_y):
                # Check for treasure
                if self.grid[new_x, new_y] == self.TREASURE:
                    reward += 10.0
                    self.treasures_collected += 1
                    self.treasure_positions = [
                        t for t in self.treasure_positions if t != [new_x, new_y]
                    ]

                x, y = new_x, new_y
                self.agent_positions[agent_id] = [x, y]
            else:
                reward -= 0.1  # Penalty for invalid move

        # Update grid with new position
        self.grid[x, y] = self.AGENT

        return reward

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid for movement"""
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        if self.grid[x, y] in [self.WALL, self.OBSTACLE, self.AGENT]:
            return False
        return True

    def _attack_adjacent_enemy(self, agent_id: int) -> float:
        """Attack enemies adjacent to agent"""
        x, y = self.agent_positions[agent_id]
        reward = 0.0
        damage = 30.0

        # Check all adjacent cells
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            # Find if there's an enemy at this position
            for i, (ex, ey) in enumerate(self.enemy_positions):
                if ex == nx and ey == ny:
                    self.enemy_healths[i] -= damage

                    if self.enemy_healths[i] <= 0:
                        # Enemy defeated
                        reward += 5.0
                        self.grid[ex, ey] = self.EMPTY
                        self.enemy_positions.pop(i)
                        self.enemy_healths.pop(i)
                    else:
                        reward += 1.0  # Partial reward for damage
                    break

        return reward

    def _update_enemies(self):
        """Simple enemy AI: move towards nearest agent and attack if adjacent"""
        for i, (ex, ey) in enumerate(self.enemy_positions):
            # Find nearest agent
            min_dist = float('inf')
            target_agent = 0
            for j, (ax, ay) in enumerate(self.agent_positions):
                dist = abs(ex - ax) + abs(ey - ay)  # Manhattan distance
                if dist < min_dist:
                    min_dist = dist
                    target_agent = j

            # Move towards target agent
            ax, ay = self.agent_positions[target_agent]

            # Check if adjacent (can attack)
            if abs(ex - ax) + abs(ey - ay) == 1:
                # Attack agent
                damage = 10.0 * (1 + 0.3 * self.difficulty)
                self.agent_healths[target_agent] -= damage
            else:
                # Move towards agent
                self.grid[ex, ey] = self.EMPTY

                if abs(ex - ax) > abs(ey - ay):
                    new_ex = ex + (1 if ax > ex else -1)
                    new_ey = ey
                else:
                    new_ex = ex
                    new_ey = ey + (1 if ay > ey else -1)

                # Check if new position is valid
                if self._is_valid_position(new_ex, new_ey):
                    self.enemy_positions[i] = [new_ex, new_ey]
                    self.grid[new_ex, new_ey] = self.ENEMY
                else:
                    # Stay in place
                    self.grid[ex, ey] = self.ENEMY

    def _calculate_rewards(self, individual_rewards: List[float]) -> List[float]:
        """
        Calculate final rewards for all agents.
        Combines individual rewards with team-based rewards.
        """
        # Team reward: bonus if all enemies defeated
        team_reward = 0.0
        if len(self.enemy_positions) == 0:
            team_reward = 20.0

        # Team reward: bonus for treasures collected
        team_reward += self.treasures_collected * 2.0

        # Penalty if any agent dies
        death_penalty = -10.0 if any(h <= 0 for h in self.agent_healths) else 0.0

        # Combine individual and team rewards
        rewards = []
        for i in range(self.num_agents):
            total_reward = individual_rewards[i] + team_reward + death_penalty
            rewards.append(total_reward)

        return rewards

    def _check_terminated(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if all agents are dead
        if all(h <= 0 for h in self.agent_healths):
            return True

        # Terminate if all enemies defeated and all treasures collected
        if len(self.enemy_positions) == 0 and self.treasures_collected == self.num_treasures:
            return True

        return False

    def _get_info(self) -> Dict:
        """Get additional info for logging"""
        return {
            'enemies_remaining': len(self.enemy_positions),
            'treasures_collected': self.treasures_collected,
            'agent_healths': self.agent_healths.copy(),
            'team_alive': sum(1 for h in self.agent_healths if h > 0),
            'success': len(self.enemy_positions) == 0 and self.treasures_collected == self.num_treasures,
        }

    def render(self, mode: str = 'human'):
        """Render the environment"""
        if mode == 'human':
            print("\n" + "="*50)
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Enemies: {len(self.enemy_positions)}, Treasures: {self.treasures_collected}/{self.num_treasures}")

            # Print grid
            symbols = {
                self.EMPTY: '.',
                self.WALL: '#',
                self.AGENT: 'A',
                self.ENEMY: 'E',
                self.TREASURE: 'T',
                self.OBSTACLE: 'X',
            }

            for i in range(self.grid_size):
                row = ""
                for j in range(self.grid_size):
                    row += symbols.get(self.grid[i, j], '?') + " "
                print(row)

            print(f"Agent Healths: {self.agent_healths}")
            print("="*50)

        elif mode == 'rgb_array':
            # Return RGB array for video rendering
            rgb = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

            # Color mapping
            colors = {
                self.EMPTY: [255, 255, 255],      # White
                self.WALL: [0, 0, 0],             # Black
                self.AGENT: [0, 255, 0],          # Green
                self.ENEMY: [255, 0, 0],          # Red
                self.TREASURE: [255, 215, 0],     # Gold
                self.OBSTACLE: [128, 128, 128],   # Gray
            }

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    rgb[i, j] = colors.get(self.grid[i, j], [255, 255, 255])

            return rgb

    def close(self):
        """Clean up resources"""
        pass
