

import torch
import numpy as np
import logging
import os
from datetime import datetime

from environments.procedural_dungeon import ProceduralDungeonEnv
from agents.transformer_policy import TransformerMetaRLPolicy
from training.curriculum_scheduler import CurriculumScheduler


def setup_logging():
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/train_meta_rl_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def train_meta_rl(
    num_episodes: int = 10000,
    num_tasks: int = 3,  # Different game domains
    task_switch_frequency: int = 1000,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
):
    """
    Train meta-RL policy with task inference.

    Args:
        num_episodes: Total training episodes
        num_tasks: Number of distinct game types/domains
        task_switch_frequency: How often to switch tasks
        learning_rate: Learning rate
        batch_size: Batch size
    """

    logger = setup_logging()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Training meta-RL with {num_tasks} tasks")

    # Create meta-RL policy
    logger.info("Creating meta-RL policy...")
    policy = TransformerMetaRLPolicy(
        obs_dim=49,  # Grid observation
        action_dim=6,
        task_latent_dim=64,
        trajectory_window=20,
        num_tasks=num_tasks,
        device=device,
    )

    # Create multiple task environments (different difficulties = different tasks)
    logger.info("Creating task environments...")
    tasks = []
    for task_id in range(num_tasks):
        difficulty = 0.2 + (task_id * 0.3)  # [0.2, 0.5, 0.8]
        env = ProceduralDungeonEnv(
            num_agents=2,
            grid_size=10,
            difficulty=difficulty,
        )
        tasks.append({'env': env, 'difficulty': difficulty, 'id': task_id})

    # Setup optimizers
    actor_optimizer = torch.optim.Adam(
        policy.task_inference.parameters(),
        lr=learning_rate
    )

    logger.info("Starting meta-RL training...")
    episode_rewards = []
    current_task = 0

    for episode in range(num_episodes):
        # Switch tasks
        if episode > 0 and episode % task_switch_frequency == 0:
            current_task = (current_task + 1) % num_tasks
            logger.info(f"Switching to task {current_task} (difficulty={tasks[current_task]['difficulty']:.1f})")

        env = tasks[current_task]['env']
        obs_list, _ = env.reset()

        episode_reward = 0.0
        episode_success = False

        # Episode loop
        for step in range(200):
            # Process trajectory (build context)
            if isinstance(obs_list[0], dict):
                obs = obs_list[0]['grid']
            else:
                obs = obs_list[0]

            # Add to trajectory buffer
            action = policy.select_action(obs, use_exploration=True)

            # Step environment
            obs_list, rewards, terminated, truncated, _ = env.step([action, 0])

            episode_reward += np.mean(rewards)

            if terminated or truncated:
                episode_success = True
                break

        episode_rewards.append(episode_reward)

        # Periodic training update
        if (episode + 1) % 10 == 0:
            policy.reset_trajectory_buffer()

        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(
                f"Episode {episode+1}/{num_episodes} | "
                f"Task {current_task} | "
                f"Avg Reward: {avg_reward:.3f}"
            )

    # Save final policy
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    policy.save(f"{checkpoint_dir}/meta_rl_final.pth")
    logger.info("Meta-RL training completed!")

    return policy


if __name__ == "__main__":
    policy = train_meta_rl(
        num_episodes=5000,
        num_tasks=3,
        task_switch_frequency=500,
    )
