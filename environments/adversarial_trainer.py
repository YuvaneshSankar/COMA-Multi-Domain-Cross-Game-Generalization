

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import logging


class AdversarialTrainer:
    """
    Trains level generator and solver agents adversarially.

    Generator creates levels.
    Solver attempts to complete levels.
    Both learn from each other's performance.
    """

    def __init__(
        self,
        generator: Any,  # LevelGenerator
        solver_agent: Any,  # COMAAgent
        coma_env: Any,  # Environment with generated levels
        num_solver_episodes: int = 10,
        num_generator_updates: int = 5,
        difficulty_range: Tuple[float, float] = (0.2, 0.8),
        max_steps: int = 200,
        device: str = 'cpu',
    ):
        """
        Initialize adversarial trainer.

        Args:
            generator: Level generator model
            solver_agent: COMA solver agent
            coma_env: Environment for solver
            num_solver_episodes: Episodes to train solver per generator update
            num_generator_updates: Number of generator updates per cycle
            difficulty_range: Range of difficulties to sample
            max_steps: Max steps per episode
            device: Torch device
        """
        self.generator = generator
        self.solver_agent = solver_agent
        self.coma_env = coma_env

        self.num_solver_episodes = num_solver_episodes
        self.num_generator_updates = num_generator_updates
        self.difficulty_range = difficulty_range
        self.max_steps = max_steps
        self.device = device

        # Tracking
        self.training_stats = {
            'solver_success_rates': deque(maxlen=1000),
            'solver_steps': deque(maxlen=1000),
            'generator_rewards': deque(maxlen=1000),
            'difficulty_progression': [],
            'episode_data': [],
        }

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('AdversarialTrainer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def train_cycle(self, cycle_num: int) -> Dict[str, float]:
        """
        One full adversarial training cycle:
        1. Sample difficulty
        2. Generate levels with sampled difficulty
        3. Solve generated levels (train solver)
        4. Compute generator reward based on solver performance
        5. Update generator
        """
        cycle_results = {
            'cycle': cycle_num,
            'avg_solver_success': 0.0,
            'avg_solver_steps': 0.0,
            'avg_generator_reward': 0.0,
            'sampled_difficulty': 0.0,
        }

        # Phase 1: Determine difficulty progression
        difficulty = self._get_curriculum_difficulty(cycle_num)
        cycle_results['sampled_difficulty'] = difficulty
        self.training_stats['difficulty_progression'].append(difficulty)

        # Phase 2: Train solver on generated levels
        solver_stats = self._train_solver_phase(difficulty, num_episodes=self.num_solver_episodes)

        cycle_results['avg_solver_success'] = solver_stats['success_rate']
        cycle_results['avg_solver_steps'] = solver_stats['avg_steps']
        cycle_results['avg_solver_health'] = solver_stats['avg_final_health']

        # Phase 3: Update generator based on solver performance
        generator_stats = self._update_generator_phase(
            difficulty,
            solver_stats,
            num_updates=self.num_generator_updates
        )

        cycle_results['avg_generator_reward'] = generator_stats['avg_reward']

        # Log cycle results
        self.logger.info(
            f"Cycle {cycle_num} | Difficulty: {difficulty:.3f} | "
            f"Solver Success: {solver_stats['success_rate']:.2%} | "
            f"Avg Steps: {solver_stats['avg_steps']:.1f} | "
            f"Generator Reward: {generator_stats['avg_reward']:.3f}"
        )

        return cycle_results

    def _get_curriculum_difficulty(self, cycle_num: int, max_cycles: int = 1000) -> float:
        """
        Get curriculum difficulty based on training progress.
        Gradually increases difficulty over training.
        """
        progress = cycle_num / max_cycles

        # Sigmoid-based progression
        min_diff, max_diff = self.difficulty_range
        difficulty = min_diff + (max_diff - min_diff) / (1.0 + np.exp(-10 * (progress - 0.5)))

        return np.clip(difficulty, min_diff, max_diff)

    def _train_solver_phase(self, difficulty: float, num_episodes: int) -> Dict[str, float]:
        """
        Train solver agent on levels generated at given difficulty.

        Returns:
            Stats about solver performance
        """
        success_count = 0
        total_steps = 0
        total_episodes = 0
        total_final_health = 0.0

        for episode in range(num_episodes):
            # Reset environment with generated level
            obs, info = self.coma_env.reset(difficulty=difficulty)

            episode_reward = 0.0
            episode_done = False
            steps = 0

            for step in range(self.max_steps):
                # Select actions
                actions = self.solver_agent.select_actions(
                    [o['position'] for o in obs],  # Extract positions as observations
                    exploration=True
                )

                # Step environment
                obs, rewards, terminated, truncated, info = self.coma_env.step(actions)

                # Store experience
                state = np.concatenate([o['position'] for o in obs])
                next_state = np.concatenate([o['position'] for o in obs])
                self.solver_agent.store_experience(
                    state, np.array(actions),
                    np.mean(rewards),
                    next_state,
                    terminated or truncated
                )

                episode_reward += np.mean(rewards)
                steps += 1

                if terminated or truncated:
                    episode_done = True
                    break

            # Check success
            success = info.get('success', False)
            if success:
                success_count += 1

            total_steps += steps
            total_final_health += info['agent_healths'][0] if len(info['agent_healths']) > 0 else 0
            total_episodes += 1

            # Train solver
            if total_episodes % 5 == 0:
                self.solver_agent.update(batch_size=32)

            # Store episode data
            self.training_stats['solver_success_rates'].append(float(success))
            self.training_stats['solver_steps'].append(steps)

        # Compute average stats
        avg_success = success_count / num_episodes if num_episodes > 0 else 0.0
        avg_steps = total_steps / num_episodes if num_episodes > 0 else 0.0
        avg_final_health = total_final_health / num_episodes if num_episodes > 0 else 0.0

        return {
            'success_rate': avg_success,
            'avg_steps': avg_steps,
            'avg_final_health': avg_final_health,
            'total_episodes': num_episodes,
        }

    def _update_generator_phase(
        self,
        difficulty: float,
        solver_stats: Dict[str, float],
        num_updates: int = 5,
    ) -> Dict[str, float]:
        """
        Update generator based on solver performance.

        Reward generator if levels are challenging but solvable.
        """
        total_reward = 0.0
        num_updates_performed = 0

        for update_idx in range(num_updates):
            # Compute reward for generator
            reward = self.generator.compute_generator_reward(
                solver_success=solver_stats['success_rate'] > 0.3,
                solver_steps=int(solver_stats['avg_steps']),
                max_steps=self.max_steps,
                solver_final_health=solver_stats['avg_final_health'],
            )

            # Update generator
            loss = self.generator.train_step(
                difficulty=difficulty,
                solver_success=solver_stats['success_rate'] > 0.3,
                solver_steps=int(solver_stats['avg_steps']),
                max_steps=self.max_steps,
                solver_final_health=solver_stats['avg_final_health'],
            )

            total_reward += reward
            num_updates_performed += 1

            self.training_stats['generator_rewards'].append(reward)

        avg_reward = total_reward / num_updates if num_updates > 0 else 0.0

        return {
            'avg_reward': avg_reward,
            'num_updates': num_updates_performed,
        }

    def train(
        self,
        num_cycles: int = 100,
        checkpoint_interval: int = 10,
        save_path: str = 'checkpoints',
    ) -> Dict[str, List]:
        """
        Run full adversarial training.

        Args:
            num_cycles: Number of adversarial cycles
            checkpoint_interval: Save checkpoint every N cycles
            save_path: Path to save checkpoints

        Returns:
            Training statistics
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        all_results = []

        for cycle in range(num_cycles):
            # Run one adversarial cycle
            cycle_result = self.train_cycle(cycle)
            all_results.append(cycle_result)

            # Periodic checkpoint
            if (cycle + 1) % checkpoint_interval == 0:
                self._save_checkpoint(f"{save_path}/cycle_{cycle}.pt")
                self.logger.info(f"Checkpoint saved at cycle {cycle}")

            # Periodic evaluation
            if (cycle + 1) % (checkpoint_interval * 2) == 0:
                eval_stats = self._evaluate()
                self.logger.info(f"Evaluation: {eval_stats}")

        return {
            'cycle_results': all_results,
            'training_stats': dict(self.training_stats),
        }

    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate current system performance.
        """
        eval_episodes = 50
        success_count = 0
        total_steps = 0

        for _ in range(eval_episodes):
            difficulty = np.mean(self.difficulty_range)
            obs, _ = self.coma_env.reset(difficulty=difficulty)

            for step in range(self.max_steps):
                actions = self.solver_agent.select_actions(
                    [o['position'] for o in obs],
                    exploration=False
                )
                obs, _, terminated, truncated, info = self.coma_env.step(actions)
                total_steps += 1

                if info.get('success', False):
                    success_count += 1

                if terminated or truncated:
                    break

        return {
            'eval_success_rate': success_count / eval_episodes,
            'eval_avg_steps': total_steps / eval_episodes,
        }

    def _save_checkpoint(self, path: str):
        """Save checkpoint"""
        checkpoint = {
            'generator_state': self.generator.generator.state_dict() if hasattr(self.generator, 'generator') else None,
            'solver_state': {
                'actors': [a.state_dict() for a in self.solver_agent.actors],
                'critic': self.solver_agent.critic.state_dict(),
            },
            'training_stats': dict(self.training_stats),
        }
        torch.save(checkpoint, path)

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'avg_solver_success': np.mean(list(self.training_stats['solver_success_rates'])) if self.training_stats['solver_success_rates'] else 0.0,
            'avg_solver_steps': np.mean(list(self.training_stats['solver_steps'])) if self.training_stats['solver_steps'] else 0.0,
            'avg_generator_reward': np.mean(list(self.training_stats['generator_rewards'])) if self.training_stats['generator_rewards'] else 0.0,
            'total_episodes_trained': len(self.training_stats['solver_success_rates']),
        }


class GeneratorSolverEnvironment:
    """
    Wrapper environment that integrates generator and solver.
    Handles the interaction between them.
    """

    def __init__(
        self,
        base_env: Any,
        generator: Any,
        num_agents: int = 2,
    ):
        self.base_env = base_env
        self.generator = generator
        self.num_agents = num_agents

    def reset(self, difficulty: float = 0.5):
        """Reset with generated level"""
        # Generate level
        level_data = self.generator.generate_level(difficulty)

        # Reset base environment
        obs, info = self.base_env.reset()

        # Apply generated level (simplified)
        # In practice, would properly integrate level_data into environment

        return obs, info

    def step(self, actions: List[int]):
        """Forward step"""
        return self.base_env.step(actions)


if __name__ == "__main__":
    print("Adversarial Training Module")
    print("Use in conjunction with generator and solver agents")
    print("\nExample:")
    print("  trainer = AdversarialTrainer(generator, solver, env)")
    print("  results = trainer.train(num_cycles=100)")
