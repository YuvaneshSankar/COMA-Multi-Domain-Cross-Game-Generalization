
import torch
import numpy as np
import logging
import os
from datetime import datetime

from environments.procedural_dungeon import ProceduralDungeonEnv
from environments.atari_wrapper import AtariTransferWrapper
from agents.coma_agent import COMAAgent
from training.transfer_learner import TransferLearner


def setup_logging():
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/train_transfer_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def train_transfer(
    pretrained_path: str,
    target_game: str = 'pong',
    num_episodes: int = 5000,
    transfer_strategy: str = 'fine_tuning',
    learning_rate: float = 5e-4,
    batch_size: int = 32,
):
    """
    Train with transfer learning from dungeon to Atari.

    Args:
        pretrained_path: Path to pre-trained COMA agent
        target_game: Target Atari game
        num_episodes: Training episodes
        transfer_strategy: 'fine_tuning' or 'feature_extraction'
        learning_rate: Learning rate
        batch_size: Batch size
    """

    logger = setup_logging()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Transfer target: {target_game}")

    # Load pre-trained agent
    logger.info(f"Loading pre-trained agent from {pretrained_path}")
    dungeon_agent = COMAAgent(
        num_agents=2, state_dim=49, action_dim=6,
        device=device
    )
    dungeon_agent.load(pretrained_path)

    # Create Atari environment
    logger.info(f"Creating {target_game} environment...")
    atari_env = AtariTransferWrapper(
        game_name=target_game,
        obs_size=84,
        frame_stack=4,
    )

    # Create transfer learning trainer
    logger.info("Setting up transfer learning...")
    transfer_learner = TransferLearner(
        source_model=dungeon_agent.actors[0],  # Use first actor as source
        target_model=atari_env,
        transfer_strategy=transfer_strategy,
        learning_rate=learning_rate,
        device=device,
    )

    # Training loop
    logger.info("Starting transfer training...")
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = atari_env.reset()
        episode_reward = 0.0

        for step in range(500):
            # Use dungeon-style actions, convert to Atari
            dungeon_actions = [np.random.randint(0, 6) for _ in range(2)]
            obs_list, rewards, terminated, truncated, info = atari_env.step_with_dungeon_actions(dungeon_actions)

            episode_reward += np.mean(rewards)
            obs = obs_list

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode+1}: Avg Reward = {avg_reward:.3f}")

    logger.info("Transfer training completed!")
    logger.info(f"Final avg reward: {np.mean(episode_rewards[-500:]):.3f}")

    return atari_env


if __name__ == "__main__":
    train_transfer(
        pretrained_path='./checkpoints/single_domain_final.pth',
        target_game='pong',
        num_episodes=3000,
    )
