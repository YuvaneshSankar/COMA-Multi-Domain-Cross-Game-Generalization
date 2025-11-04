
import torch
import numpy as np
import logging
from datetime import datetime
import os

# Import your custom modules
from environments.procedural_dungeon import ProceduralDungeonEnv
from agents.coma_agent import COMAAgent
from training.curriculum_scheduler import CurriculumScheduler, DifficultyAdapterOnline


def setup_logging():
    """Setup logging"""
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/train_single_domain_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def train_single_domain(
    num_episodes: int = 10000,
    num_agents: int = 2,
    initial_grid_size: int = 8,
    max_grid_size: int = 14,
    difficulty_schedule: str = 'sigmoid',
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    save_interval: int = 100,
):
    """
    Main training loop for single domain.

    Args:
        num_episodes: Total training episodes
        num_agents: Number of agents
        initial_grid_size: Starting grid size
        max_grid_size: Maximum grid size
        difficulty_schedule: How difficulty progresses
        learning_rate: Learning rate
        batch_size: Batch size for updates
        save_interval: Save checkpoint every N episodes
    """

    logger = setup_logging()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create environment
    logger.info("Creating environment...")
    env = ProceduralDungeonEnv(
        num_agents=num_agents,
        grid_size=initial_grid_size,
        num_enemies=2,
        num_treasures=1,
        num_obstacles=4,
        difficulty=0.0,
        vision_range=3,
        max_steps=200,
    )

    # Create COMA agent
    logger.info("Creating COMA agent...")
    state_dim = 49  # (2*3+1)^2 for vision_range=3
    agent = COMAAgent(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=6,
        hidden_dim=128,
        learning_rate=learning_rate,
        device=device,
    )

    # Create curriculum scheduler
    scheduler = CurriculumScheduler(
        total_episodes=num_episodes,
        num_phases=3,
        difficulty_schedule=difficulty_schedule,
    )

    # Training loop
    episode_rewards = []
    success_count = 0
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Starting training...")

    for episode in range(num_episodes):
        # Update schedule
        schedule_info = scheduler.step()
        difficulty = schedule_info['difficulty']

        # Update environment difficulty
        env.difficulty = difficulty

        # Update grid size progressively
        progress = schedule_info['overall_progress']
        grid_size = int(initial_grid_size + (max_grid_size - initial_grid_size) * progress)
        env.grid_size = grid_size

        # Reset environment
        obs, info = env.reset()
        episode_reward = 0.0
        episode_success = False

        # Episode loop
        for step in range(env.max_steps):
            # Select actions
            actions = agent.select_actions(obs, exploration=True)

            # Step environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            # Store experience
            state = np.concatenate([o['position'] for o in obs])
            next_state = np.concatenate([o['position'] for o in next_obs])
            agent.store_experience(
                state, np.array(actions),
                np.mean(rewards),
                next_state,
                terminated or truncated
            )

            episode_reward += np.mean(rewards)
            obs = next_obs

            if info.get('success', False):
                episode_success = True

            if terminated or truncated:
                break

        # Train agent
        if episode % 5 == 0:
            losses = agent.update(batch_size=batch_size)

        # Track metrics
        episode_rewards.append(episode_reward)
        if episode_success:
            success_count += 1

        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            recent_success = success_count / 50 if episode >= 50 else success_count / (episode + 1)

            logger.info(
                f"Episode {episode+1}/{num_episodes} | "
                f"Difficulty: {difficulty:.3f} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Success Rate: {recent_success:.2%} | "
                f"Grid Size: {grid_size}x{grid_size}"
            )

            success_count = 0

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/single_domain_ep{episode+1}.pth"
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Final save
    final_path = f"{checkpoint_dir}/single_domain_final.pth"
    agent.save(final_path)
    logger.info(f"Final model saved: {final_path}")

    logger.info("Training completed!")

    return agent, env


if __name__ == "__main__":
    agent, env = train_single_domain(
        num_episodes=5000,
        num_agents=2,
        difficulty_schedule='sigmoid',
    )
