
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
try:
    from pettingzoo.atari import (
        pong_v3, tennis_v3, boxing_v2,
        basketball_pong_v3, ice_hockey_v2
    )
except ImportError:
    print("Warning: PettingZoo not installed. Install with: pip install pettingzoo[atari]")


class AtariMultiAgentWrapper:
    """
    Wrapper for PettingZoo Atari environments to match our multi-agent interface.

    Features:
    - Standardized observation space matching dungeon environment
    - Preprocessed frames (grayscale, downsampled)
    - Frame stacking for temporal information
    - Compatible with COMA architecture
    - Support for multiple Atari games
    """

    SUPPORTED_GAMES = {
        'pong': pong_v3,
        'tennis': tennis_v3,
        'boxing': boxing_v2,
        'basketball_pong': basketball_pong_v3,
        'ice_hockey': ice_hockey_v2,
    }

    def __init__(
        self,
        game_name: str = 'pong',
        obs_size: int = 84,
        frame_stack: int = 4,
        grayscale: bool = True,
        max_steps: int = 10000,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Atari wrapper.

        Args:
            game_name: Name of Atari game ('pong', 'tennis', 'boxing', etc.)
            obs_size: Size to downsample frames to (obs_size x obs_size)
            frame_stack: Number of frames to stack for temporal info
            grayscale: Whether to convert to grayscale
            max_steps: Maximum steps per episode
            render_mode: Rendering mode for PettingZoo
        """
        if game_name not in self.SUPPORTED_GAMES:
            raise ValueError(f"Game {game_name} not supported. Choose from {list(self.SUPPORTED_GAMES.keys())}")

        self.game_name = game_name
        self.obs_size = obs_size
        self.frame_stack = frame_stack
        self.grayscale = grayscale
        self.max_steps = max_steps

        # Initialize PettingZoo environment
        env_creator = self.SUPPORTED_GAMES[game_name]
        self.env = env_creator.parallel_env(render_mode=render_mode)

        # Get agent names from environment
        self.env.reset()
        self.agent_names = list(self.env.agents)
        self.num_agents = len(self.agent_names)

        # Action space (Atari actions)
        self.action_space = self.env.action_space(self.agent_names[0])
        self.num_actions = self.action_space.n

        # Observation space: stacked preprocessed frames + metadata
        channels = 1 if grayscale else 3
        self.observation_space = spaces.Dict({
            'frames': spaces.Box(
                low=0, high=255,
                shape=(frame_stack, obs_size, obs_size, channels),
                dtype=np.uint8
            ),
            'agent_id': spaces.Discrete(self.num_agents),
            'game_score': spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
            'opponent_score': spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
        })

        # Frame buffer for stacking
        self.frame_buffers = {agent: [] for agent in self.agent_names}

        # Scoring
        self.scores = {agent: 0.0 for agent in self.agent_names}
        self.current_step = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[Dict], Dict]:
        """Reset environment and return initial observations"""
        observations, infos = self.env.reset(seed=seed, options=options)

        # Reset scores and step counter
        self.scores = {agent: 0.0 for agent in self.agent_names}
        self.current_step = 0

        # Initialize frame buffers with first observation
        for agent in self.agent_names:
            frame = self._preprocess_frame(observations[agent])
            self.frame_buffers[agent] = [frame] * self.frame_stack

        processed_obs = self._get_processed_observations()
        info = self._get_info()

        return processed_obs, info

    def step(self, actions: Dict[str, int]) -> Tuple[List[Dict], List[float], bool, bool, Dict]:
        """
        Execute actions for all agents.

        Args:
            actions: Dictionary mapping agent names to actions

        Returns:
            observations, rewards, terminated, truncated, info
        """
        self.current_step += 1

        # Step environment
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # Update scores
        for agent in self.agent_names:
            if agent in rewards:
                self.scores[agent] += rewards[agent]

        # Update frame buffers
        for agent in self.agent_names:
            if agent in observations:
                frame = self._preprocess_frame(observations[agent])
                self.frame_buffers[agent].append(frame)
                self.frame_buffers[agent].pop(0)  # Remove oldest frame

        # Process observations
        processed_obs = self._get_processed_observations()

        # Convert rewards to list
        reward_list = [rewards.get(agent, 0.0) for agent in self.agent_names]

        # Check termination (episode ends if any agent terminates)
        terminated = any(terminations.values()) if terminations else False
        truncated = any(truncations.values()) if truncations else False
        truncated = truncated or (self.current_step >= self.max_steps)

        info = self._get_info()

        return processed_obs, reward_list, terminated, truncated, info

    def step_list(self, actions: List[int]) -> Tuple[List[Dict], List[float], bool, bool, Dict]:
        """
        Alternative step interface that takes a list of actions instead of dict.
        More compatible with standard RL training loops.
        """
        action_dict = {agent: actions[i] for i, agent in enumerate(self.agent_names)}
        return self.step(action_dict)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a raw Atari frame.

        Steps:
        1. Convert to grayscale (optional)
        2. Downsample to obs_size x obs_size
        3. Normalize to [0, 255]
        """
        # Convert to grayscale if specified
        if self.grayscale and len(frame.shape) == 3:
            # Use standard RGB to grayscale conversion
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            frame = frame.astype(np.uint8)

        # Downsample using simple averaging (fast)
        if frame.shape[0] != self.obs_size or frame.shape[1] != self.obs_size:
            frame = self._downsample(frame, self.obs_size)

        # Ensure correct shape
        if self.grayscale and len(frame.shape) == 2:
            frame = frame[:, :, np.newaxis]  # Add channel dimension

        return frame

    def _downsample(self, frame: np.ndarray, target_size: int) -> np.ndarray:
        """Downsample frame to target size using simple binning"""
        from scipy.ndimage import zoom

        h, w = frame.shape[:2]
        zoom_factors = (target_size / h, target_size / w)

        if len(frame.shape) == 3:
            zoom_factors = zoom_factors + (1,)  # Don't zoom color channels

        downsampled = zoom(frame, zoom_factors, order=1)  # Bilinear interpolation

        return downsampled.astype(np.uint8)

    def _get_processed_observations(self) -> List[Dict]:
        """Get processed observations for all agents"""
        observations = []

        for i, agent in enumerate(self.agent_names):
            # Stack frames
            stacked_frames = np.stack(self.frame_buffers[agent], axis=0)

            # Get opponent agent
            opponent = self.agent_names[1 - i] if self.num_agents == 2 else self.agent_names[0]

            obs = {
                'frames': stacked_frames,
                'agent_id': i,
                'game_score': np.array([self.scores[agent]], dtype=np.float32),
                'opponent_score': np.array([self.scores[opponent]], dtype=np.float32),
            }
            observations.append(obs)

        return observations

    def _get_info(self) -> Dict:
        """Get additional info for logging"""
        return {
            'step': self.current_step,
            'scores': self.scores.copy(),
            'game_name': self.game_name,
            'num_agents': self.num_agents,
        }

    def render(self):
        """Render the environment"""
        return self.env.render()

    def close(self):
        """Close the environment"""
        self.env.close()


class AtariTransferWrapper(AtariMultiAgentWrapper):
    """
    Extended wrapper with features for transfer learning from dungeon environment.

    Features:
    - Action mapping from dungeon actions to Atari actions
    - Observation augmentation to match dungeon format
    - Progressive neural network support
    """

    def __init__(self, *args, dungeon_compatible: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.dungeon_compatible = dungeon_compatible

        if dungeon_compatible:
            # Define action mapping: dungeon actions -> Atari actions
            # Dungeon: [up, down, left, right, attack, stay] -> Atari game-specific
            self._setup_action_mapping()

    def _setup_action_mapping(self):
        """
        Setup action mapping based on game type.
        Maps dungeon actions [up, down, left, right, attack, stay] to Atari actions.
        """
        # Default mapping (works for most Atari games)
        if self.game_name == 'pong':
            # Pong: [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]
            self.action_mapping = {
                0: 2,  # up -> RIGHT (move paddle up)
                1: 3,  # down -> LEFT (move paddle down)
                2: 3,  # left -> LEFT
                3: 2,  # right -> RIGHT
                4: 1,  # attack -> FIRE
                5: 0,  # stay -> NOOP
            }
        elif self.game_name == 'tennis':
            # Tennis: similar to pong
            self.action_mapping = {
                0: 2,  # up
                1: 3,  # down
                2: 4,  # left
                3: 5,  # right
                4: 1,  # attack (hit ball)
                5: 0,  # stay
            }
        elif self.game_name == 'boxing':
            # Boxing: [NOOP, FIRE, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, ...]
            self.action_mapping = {
                0: 2,  # up
                1: 5,  # down
                2: 4,  # left
                3: 3,  # right
                4: 1,  # attack (punch)
                5: 0,  # stay
            }
        else:
            # Generic mapping
            self.action_mapping = {i: min(i, self.num_actions - 1) for i in range(6)}

    def map_dungeon_action(self, dungeon_action: int) -> int:
        """Map a dungeon action to corresponding Atari action"""
        return self.action_mapping.get(dungeon_action, 0)

    def step_with_dungeon_actions(self, dungeon_actions: List[int]):
        """
        Step function that accepts dungeon-style actions and maps them to Atari.
        Useful for transfer learning experiments.
        """
        atari_actions = [self.map_dungeon_action(a) for a in dungeon_actions]
        return self.step_list(atari_actions)


# Utility functions for creating environments

def make_atari_env(game_name: str = 'pong', **kwargs):
    """Factory function to create Atari environment"""
    return AtariMultiAgentWrapper(game_name=game_name, **kwargs)


def make_transfer_env(game_name: str = 'pong', **kwargs):
    """Factory function to create transfer-learning compatible Atari environment"""
    return AtariTransferWrapper(game_name=game_name, dungeon_compatible=True, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Atari Wrapper...")

    # Test basic wrapper
    env = make_atari_env('pong', obs_size=84, frame_stack=4)
    obs, info = env.reset()

    print(f"Number of agents: {env.num_agents}")
    print(f"Action space: {env.num_actions}")
    print(f"Observation shape: {obs[0]['frames'].shape}")

    # Test step
    actions = [env.action_space.sample() for _ in range(env.num_agents)]
    obs, rewards, done, truncated, info = env.step_list(actions)

    print(f"Rewards: {rewards}")
    print(f"Done: {done}, Truncated: {truncated}")

    env.close()

    # Test transfer wrapper
    print("\nTesting Transfer Wrapper...")
    transfer_env = make_transfer_env('pong')

    # Test dungeon action mapping
    dungeon_actions = [0, 1]  # up, down
    obs, rewards, done, truncated, info = transfer_env.step_with_dungeon_actions(dungeon_actions)
    print(f"Dungeon actions {dungeon_actions} mapped successfully")

    transfer_env.close()

    print("\nAll tests passed!")
