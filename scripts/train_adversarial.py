

import torch
import numpy as np
import logging
import os
from datetime import datetime

from environments.procedural_dungeon import ProceduralDungeonEnv
from environments.level_generator import AdversarialLevelGenerator, GeneratedDungeonEnv
from agents.coma_agent import COMAAgent
from training.adversarial_trainer import AdversarialTrainer


def setup_logging():
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/train_adversarial_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def train_adversarial(
    pretrained_solver_path: str = None,
    num_cycles: int = 100,
    num_solver_episodes: int = 20,
    num_generator_updates: int = 5,
    difficulty_range: tuple = (0.3, 0.8),
):
    """
    Adversarial training of generator and solver.

    Args:
        pretrained_solver_path: Path to pre-trained solver (optional)
        num_cycles: Number of adversarial cycles
        num_solver_episodes: Episodes solver trains per cycle
        num_generator_updates: Generator updates per cycle
        difficulty_range: Range of difficulties to sample
    """

    logger = setup_logging()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info("Starting adversarial training...")

    # Create base environment
    base_env = ProceduralDungeonEnv(
        num_agents=2,
        grid_size=10,
        difficulty=0.5,
    )

    # Create level generator
    logger.info("Creating level generator...")
    generator = AdversarialLevelGenerator(
        grid_size=10,
        latent_dim=128,
        device=device,
    )

    # Create solver agent
    logger.info("Creating COMA solver...")
    solver = COMAAgent(
        num_agents=2,
        state_dim=49,
        action_dim=6,
        device=device,
    )

    if pretrained_solver_path:
        logger.info(f"Loading pre-trained solver from {pretrained_solver_path}")
        solver.load(pretrained_solver_path)

    # Create environment with generator
    generated_env = GeneratedDungeonEnv(
        base_env=base_env,
        generator=generator,
    )

    # Create adversarial trainer
    logger.info("Creating adversarial trainer...")
    trainer = AdversarialTrainer(
        generator=generator,
        solver_agent=solver,
        coma_env=generated_env,
        num_solver_episodes=num_solver_episodes,
        num_generator_updates=num_generator_updates,
        difficulty_range=difficulty_range,
        max_steps=200,
        device=device,
    )

    # Run adversarial training
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Starting adversarial cycles...")
    results = trainer.train(
        num_cycles=num_cycles,
        checkpoint_interval=10,
        save_path=checkpoint_dir,
    )

    # Save final models
    logger.info("Saving final models...")
    generator.save(f"{checkpoint_dir}/generator_final.pth")
    solver.save(f"{checkpoint_dir}/solver_adversarial_final.pth")

    logger.info("Adversarial training completed!")

    return generator, solver, results


if __name__ == "__main__":
    generator, solver, results = train_adversarial(
        num_cycles=50,
        num_solver_episodes=15,
    )
