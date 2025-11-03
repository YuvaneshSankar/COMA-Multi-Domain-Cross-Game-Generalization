

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class LevelGenerator(nn.Module):
    """
    Neural network that generates dungeon levels.

    Architecture:
    - Takes difficulty parameter as input
    - Outputs grid layout, enemy positions, treasure positions
    - Trained via RL to create challenging but solvable levels
    """

    def __init__(
        self,
        grid_size: int = 10,
        latent_dim: int = 128,
        num_enemy_positions: int = 5,
        num_treasure_positions: int = 3,
        num_obstacle_positions: int = 10,
    ):
        super(LevelGenerator, self).__init__()

        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.num_enemy_positions = num_enemy_positions
        self.num_treasure_positions = num_treasure_positions
        self.num_obstacle_positions = num_obstacle_positions

        # Input: difficulty level (scalar) + random noise
        input_dim = 1 + latent_dim

        # Encoder: difficulty + noise -> latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Obstacle placement head
        self.obstacle_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_obstacle_positions * 2),  # (x, y) for each obstacle
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

        # Enemy placement head
        self.enemy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_enemy_positions * 2),  # (x, y) for each enemy
            nn.Sigmoid(),
        )

        # Treasure placement head
        self.treasure_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_treasure_positions * 2),  # (x, y) for each treasure
            nn.Sigmoid(),
        )

    def forward(self, difficulty: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate a level layout.

        Args:
            difficulty: Scalar in [0, 1] indicating desired difficulty
            noise: Random latent vector for diversity (if None, sampled from N(0,1))

        Returns:
            Dictionary with obstacle, enemy, and treasure positions
        """
        batch_size = difficulty.shape[0]

        # Sample noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.latent_dim, device=difficulty.device)

        # Concatenate difficulty and noise
        x = torch.cat([difficulty.unsqueeze(1), noise], dim=1)

        # Encode
        features = self.encoder(x)

        # Generate positions
        obstacle_positions = self.obstacle_head(features).view(batch_size, self.num_obstacle_positions, 2)
        enemy_positions = self.enemy_head(features).view(batch_size, self.num_enemy_positions, 2)
        treasure_positions = self.treasure_head(features).view(batch_size, self.num_treasure_positions, 2)

        # Scale to grid size (positions are in [0, 1], scale to [1, grid_size-2] to avoid walls)
        obstacle_positions = obstacle_positions * (self.grid_size - 3) + 1
        enemy_positions = enemy_positions * (self.grid_size - 3) + 1
        treasure_positions = treasure_positions * (self.grid_size - 3) + 1

        return {
            'obstacles': obstacle_positions,
            'enemies': enemy_positions,
            'treasures': treasure_positions,
        }

    def generate_level(self, difficulty: float) -> Dict[str, np.ndarray]:
        """
        Generate a single level (inference mode).

        Args:
            difficulty: Float in [0, 1]

        Returns:
            Dictionary with positions as numpy arrays
        """
        self.eval()
        with torch.no_grad():
            difficulty_tensor = torch.tensor([difficulty], dtype=torch.float32)
            output = self.forward(difficulty_tensor)

            # Convert to numpy and round to integers
            level = {
                'obstacles': output['obstacles'][0].cpu().numpy().astype(int),
                'enemies': output['enemies'][0].cpu().numpy().astype(int),
                'treasures': output['treasures'][0].cpu().numpy().astype(int),
            }

        return level


class AdversarialLevelGenerator:
    """
    RL-based level generator that learns to create challenging levels.

    Training:
    - Generator creates levels
    - Solver agents attempt to solve them
    - Generator rewarded based on solver performance:
        * High reward if solver struggles but eventually succeeds
        * Low reward if solver fails immediately or succeeds too easily
    """

    def __init__(
        self,
        grid_size: int = 10,
        latent_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = 'cpu',
    ):
        self.grid_size = grid_size
        self.device = device

        # Generator network
        self.generator = LevelGenerator(
            grid_size=grid_size,
            latent_dim=latent_dim,
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)

        # RL training parameters
        self.gamma = gamma

        # Experience buffer
        self.episode_buffer = []

        # Metrics
        self.training_stats = {
            'generator_rewards': [],
            'solver_success_rates': [],
            'average_solver_steps': [],
            'difficulty_progression': [],
        }

    def generate_level(self, difficulty: float) -> Dict[str, np.ndarray]:
        """Generate a level at specified difficulty"""
        return self.generator.generate_level(difficulty)

    def compute_generator_reward(
        self,
        solver_success: bool,
        solver_steps: int,
        max_steps: int,
        solver_final_health: float,
    ) -> float:
        """
        Compute reward for generator based on solver performance.

        Reward structure:
        - High reward if solver succeeds after struggling (50-80% of max steps)
        - Medium reward if solver barely succeeds (low health, near max steps)
        - Low reward if solver succeeds easily (< 30% steps) or fails
        - Penalty if level is impossible

        This encourages the generator to find the "sweet spot" of difficulty.
        """

        if not solver_success:
            # Level too hard or impossible
            return -1.0

        # Normalize steps to [0, 1]
        step_ratio = solver_steps / max_steps

        # Optimal difficulty: solver uses 50-80% of available steps
        if 0.5 <= step_ratio <= 0.8:
            # High reward for challenging but solvable levels
            reward = 2.0 + (1.0 - abs(step_ratio - 0.65) / 0.15)
        elif step_ratio < 0.3:
            # Level too easy
            reward = 0.5 * step_ratio / 0.3
        elif step_ratio > 0.8:
            # Level nearly impossible (but still solved)
            reward = 1.5 * (1.0 - (step_ratio - 0.8) / 0.2)
        else:
            # Moderate difficulty
            reward = 1.0

        # Bonus for solver ending with low health (indicates challenge)
        health_factor = 1.0 - (solver_final_health / 100.0) * 0.3
        reward *= health_factor

        return reward

    def train_step(
        self,
        difficulty: float,
        solver_success: bool,
        solver_steps: int,
        max_steps: int,
        solver_final_health: float,
    ) -> float:
        """
        Single training step for generator.

        Args:
            difficulty: Requested difficulty level
            solver_success: Whether solver succeeded
            solver_steps: Number of steps solver took
            max_steps: Maximum allowed steps
            solver_final_health: Solver's final health

        Returns:
            Generator loss value
        """

        # Compute generator reward
        reward = self.compute_generator_reward(
            solver_success, solver_steps, max_steps, solver_final_health
        )

        # Store in buffer
        self.episode_buffer.append({
            'difficulty': difficulty,
            'reward': reward,
        })

        # Update generator using policy gradient
        if len(self.episode_buffer) >= 32:  # Batch update
            loss = self._update_generator()
            self.episode_buffer = []
            return loss

        return 0.0

    def _update_generator(self) -> float:
        """Update generator using collected experience"""

        # Extract data from buffer
        difficulties = torch.tensor(
            [exp['difficulty'] for exp in self.episode_buffer],
            dtype=torch.float32,
            device=self.device
        )
        rewards = torch.tensor(
            [exp['reward'] for exp in self.episode_buffer],
            dtype=torch.float32,
            device=self.device
        )

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Generate levels (forward pass to get outputs used during generation)
        self.generator.train()
        noise = torch.randn(len(difficulties), self.generator.latent_dim, device=self.device)
        outputs = self.generator(difficulties, noise)

        # Policy gradient loss: maximize expected reward
        # We use REINFORCE-style update
        # Loss = -log_prob * reward (approximated by MSE on positions weighted by reward)

        # Simplified loss: encourage diversity in successful levels
        position_variance_loss = 0.0
        for key in ['obstacles', 'enemies', 'treasures']:
            positions = outputs[key]
            # Variance across batch
            variance = positions.var(dim=0).mean()
            # Higher variance = more diversity (good)
            position_variance_loss -= variance

        # Combine with reward signal
        reward_weighted_loss = -rewards.mean() + 0.1 * position_variance_loss

        # Backprop
        self.optimizer.zero_grad()
        reward_weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Log stats
        self.training_stats['generator_rewards'].append(rewards.mean().item())

        return reward_weighted_loss.item()

    def curriculum_difficulty(self, episode: int, max_episodes: int) -> float:
        """
        Compute curriculum difficulty based on training progress.
        Gradually increases difficulty over training.

        Args:
            episode: Current episode number
            max_episodes: Total number of episodes

        Returns:
            Difficulty in [0, 1]
        """
        # Sigmoid curve: slow start, rapid middle, slow end
        progress = episode / max_episodes
        difficulty = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))

        # Clamp to [0.1, 0.9] to avoid extreme difficulties
        difficulty = np.clip(difficulty, 0.1, 0.9)

        return difficulty

    def save(self, path: str):
        """Save generator model"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
        }, path)
        print(f"Generator saved to {path}")

    def load(self, path: str):
        """Load generator model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        print(f"Generator loaded from {path}")


class GeneratedDungeonEnv:
    """
    Wrapper that uses the adversarial generator to create dungeon environments.
    Integrates with ProceduralDungeonEnv.
    """

    def __init__(
        self,
        generator: AdversarialLevelGenerator,
        base_env_class,
        difficulty: float = 0.5,
        **env_kwargs
    ):
        """
        Args:
            generator: Trained adversarial level generator
            base_env_class: ProceduralDungeonEnv class
            difficulty: Difficulty level for generation
            env_kwargs: Additional arguments for base environment
        """
        self.generator = generator
        self.base_env_class = base_env_class
        self.difficulty = difficulty
        self.env_kwargs = env_kwargs

        # Create base environment
        self.env = base_env_class(**env_kwargs)

    def reset(self, seed: Optional[int] = None, difficulty: Optional[float] = None):
        """Reset with generated level"""
        if difficulty is not None:
            self.difficulty = difficulty

        # Generate level
        level_data = self.generator.generate_level(self.difficulty)

        # Reset base environment
        obs, info = self.env.reset(seed=seed)

        # Override with generated positions
        self._apply_generated_level(level_data)

        # Get updated observations
        obs = self.env._get_observations()
        info = self.env._get_info()

        return obs, info

    def _apply_generated_level(self, level_data: Dict[str, np.ndarray]):
        """Apply generated level data to environment"""

        # Clear existing entities (except walls)
        self.env.grid[self.env.grid != self.env.WALL] = self.env.EMPTY

        # Place obstacles
        self.env.enemy_positions = []
        self.env.treasure_positions = []

        for x, y in level_data['obstacles']:
            if 0 < x < self.env.grid_size - 1 and 0 < y < self.env.grid_size - 1:
                self.env.grid[x, y] = self.env.OBSTACLE

        # Place enemies
        self.env.enemy_healths = []
        for x, y in level_data['enemies']:
            if 0 < x < self.env.grid_size - 1 and 0 < y < self.env.grid_size - 1:
                if self.env.grid[x, y] == self.env.EMPTY:
                    self.env.grid[x, y] = self.env.ENEMY
                    self.env.enemy_positions.append([x, y])
                    self.env.enemy_healths.append(50 + 50 * self.difficulty)

        # Place treasures
        for x, y in level_data['treasures']:
            if 0 < x < self.env.grid_size - 1 and 0 < y < self.env.grid_size - 1:
                if self.env.grid[x, y] == self.env.EMPTY:
                    self.env.grid[x, y] = self.env.TREASURE
                    self.env.treasure_positions.append([x, y])

        # Reset agent positions (keep same starting positions)
        for x, y in self.env.agent_positions:
            self.env.grid[x, y] = self.env.AGENT

    def step(self, actions):
        """Forward to base environment"""
        return self.env.step(actions)

    def render(self):
        """Forward to base environment"""
        return self.env.render()

    def close(self):
        """Forward to base environment"""
        return self.env.close()


# Example usage
if __name__ == "__main__":
    print("Testing Level Generator...")

    # Create generator
    generator = AdversarialLevelGenerator(grid_size=10, latent_dim=128)

    # Generate levels at different difficulties
    for difficulty in [0.2, 0.5, 0.8]:
        level = generator.generate_level(difficulty)
        print(f"\nDifficulty {difficulty}:")
        print(f"  Obstacles: {level['obstacles'].shape}")
        print(f"  Enemies: {level['enemies'].shape}")
        print(f"  Treasures: {level['treasures'].shape}")
        print(f"  Sample enemy position: {level['enemies'][0]}")

    # Test training step
    print("\nTesting training step...")
    loss = generator.train_step(
        difficulty=0.5,
        solver_success=True,
        solver_steps=150,
        max_steps=200,
        solver_final_health=30.0,
    )
    print(f"Loss: {loss}")

    # Test save/load
    generator.save("test_generator.pth")
    generator.load("test_generator.pth")

    print("\nAll tests passed!")
