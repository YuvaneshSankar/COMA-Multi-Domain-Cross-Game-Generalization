

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class GeneralizationTester:
    """
    Comprehensive generalization testing suite.
    Tests on unseen level distributions, new game types, etc.
    """

    def __init__(
        self,
        agent: Any,
        environments: Dict[str, Any],
        num_test_episodes: int = 100,
        max_steps: int = 200,
    ):
        """
        Initialize generalization tester.

        Args:
            agent: Trained agent to test
            environments: Dict of test environments
            num_test_episodes: Episodes per test
            max_steps: Max steps per episode
        """
        self.agent = agent
        self.environments = environments
        self.num_test_episodes = num_test_episodes
        self.max_steps = max_steps

        self.results = defaultdict(list)

    def test_distribution_shift(self, env_name: str) -> Dict[str, float]:
        """
        Test on different difficulty distributions.

        Args:
            env_name: Environment to test on

        Returns:
            Performance metrics under distribution shift
        """
        env = self.environments[env_name]

        results = {
            'success_rates': [],
            'avg_steps': [],
            'difficulties_tested': [],
        }

        # Test across difficulty range
        for difficulty in np.linspace(0.0, 1.0, 11):
            successes = 0
            total_steps = 0

            for episode in range(self.num_test_episodes):
                obs, _ = env.reset(difficulty=difficulty)

                for step in range(self.max_steps):
                    # Greedy action (no exploration)
                    if hasattr(self.agent, 'select_actions'):
                        actions = self.agent.select_actions(obs, exploration=False)
                    else:
                        actions = self.agent.select_action(obs, exploration=False)

                    obs, _, terminated, truncated, info = env.step(actions)
                    total_steps += 1

                    if info.get('success', False):
                        successes += 1

                    if terminated or truncated:
                        break

            success_rate = successes / self.num_test_episodes
            avg_steps = total_steps / self.num_test_episodes

            results['success_rates'].append(success_rate)
            results['avg_steps'].append(avg_steps)
            results['difficulties_tested'].append(difficulty)

        return results

    def test_unseen_seeds(self, env_name: str) -> Dict[str, float]:
        """
        Test on completely new random seeds (unseen layouts).

        Args:
            env_name: Environment to test on

        Returns:
            Performance on unseen random instances
        """
        env = self.environments[env_name]

        successes = 0
        total_steps = 0
        episode_rewards = []

        # Test on new random seeds
        for episode in range(self.num_test_episodes):
            seed = 100000 + episode  # Use seeds never seen in training
            obs, _ = env.reset(seed=seed)

            episode_reward = 0.0
            success = False

            for step in range(self.max_steps):
                if hasattr(self.agent, 'select_actions'):
                    actions = self.agent.select_actions(obs, exploration=False)
                else:
                    actions = self.agent.select_action(obs, exploration=False)

                obs, rewards, terminated, truncated, info = env.step(actions)

                if isinstance(rewards, list):
                    episode_reward += np.mean(rewards)
                else:
                    episode_reward += rewards

                total_steps += 1

                if info.get('success', False):
                    successes += 1
                    success = True

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)

        success_rate = successes / self.num_test_episodes
        avg_steps = total_steps / self.num_test_episodes
        avg_reward = np.mean(episode_rewards)

        return {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
            'std_reward': np.std(episode_rewards),
        }

    def test_domain_transfer(
        self,
        source_env_name: str,
        target_env_name: str,
    ) -> Dict[str, float]:
        """
        Test transfer from one domain to another.

        Args:
            source_env_name: Environment agent was trained on
            target_env_name: New environment to test on

        Returns:
            Transfer performance metrics
        """
        target_env = self.environments[target_env_name]

        successes = 0
        total_steps = 0

        for episode in range(self.num_test_episodes):
            obs, _ = target_env.reset()

            for step in range(self.max_steps):
                if hasattr(self.agent, 'select_actions'):
                    actions = self.agent.select_actions(obs, exploration=False)
                else:
                    actions = self.agent.select_action(obs, exploration=False)

                obs, _, terminated, truncated, info = target_env.step(actions)
                total_steps += 1

                if info.get('success', False):
                    successes += 1

                if terminated or truncated:
                    break

        return {
            'transfer_success_rate': successes / self.num_test_episodes,
            'transfer_avg_steps': total_steps / self.num_test_episodes,
        }

    def test_robustness(
        self,
        env_name: str,
        perturbation_types: List[str] = ['noise', 'missing_observations'],
    ) -> Dict[str, Dict]:
        """
        Test robustness to perturbations.

        Args:
            env_name: Environment to test
            perturbation_types: Types of perturbations to apply

        Returns:
            Robustness metrics for each perturbation
        """
        env = self.environments[env_name]

        robustness_results = {}

        for perturbation in perturbation_types:
            successes = 0

            for episode in range(self.num_test_episodes):
                obs, _ = env.reset()

                for step in range(self.max_steps):
                    # Apply perturbation
                    perturbed_obs = self._apply_perturbation(obs, perturbation)

                    if hasattr(self.agent, 'select_actions'):
                        actions = self.agent.select_actions(perturbed_obs, exploration=False)
                    else:
                        actions = self.agent.select_action(perturbed_obs, exploration=False)

                    obs, _, terminated, truncated, info = env.step(actions)

                    if info.get('success', False):
                        successes += 1

                    if terminated or truncated:
                        break

            robustness_results[perturbation] = {
                'success_rate': successes / self.num_test_episodes,
            }

        return robustness_results

    def _apply_perturbation(self, obs: Any, perturbation_type: str) -> Any:
        """Apply perturbation to observations"""
        if isinstance(obs, list):
            obs_copy = [o.copy() if isinstance(o, np.ndarray) else o for o in obs]
        else:
            obs_copy = obs.copy() if isinstance(obs, np.ndarray) else obs

        if perturbation_type == 'noise':
            # Add Gaussian noise
            if isinstance(obs_copy, list):
                for i in range(len(obs_copy)):
                    if isinstance(obs_copy[i], np.ndarray):
                        obs_copy[i] += np.random.normal(0, 0.1, obs_copy[i].shape)
            else:
                obs_copy += np.random.normal(0, 0.1, obs_copy.shape)

        elif perturbation_type == 'missing_observations':
            # Randomly zero out observations
            if isinstance(obs_copy, list):
                for i in range(len(obs_copy)):
                    if isinstance(obs_copy[i], np.ndarray):
                        mask = np.random.random(obs_copy[i].shape) > 0.8
                        obs_copy[i][mask] = 0
            else:
                mask = np.random.random(obs_copy.shape) > 0.8
                obs_copy[mask] = 0

        return obs_copy

    def test_extrapolation(
        self,
        env_name: str,
        test_difficulties: List[float],
    ) -> Dict[str, float]:
        """
        Test extrapolation to difficulties beyond training range.

        Args:
            env_name: Environment to test
            test_difficulties: Difficulties to test (may be > 1.0)

        Returns:
            Extrapolation performance
        """
        env = self.environments[env_name]

        results = {}

        for difficulty in test_difficulties:
            successes = 0

            for episode in range(self.num_test_episodes):
                obs, _ = env.reset(difficulty=np.clip(difficulty, 0, 1))

                for step in range(self.max_steps):
                    if hasattr(self.agent, 'select_actions'):
                        actions = self.agent.select_actions(obs, exploration=False)
                    else:
                        actions = self.agent.select_action(obs, exploration=False)

                    obs, _, terminated, truncated, info = env.step(actions)

                    if info.get('success', False):
                        successes += 1

                    if terminated or truncated:
                        break

            results[f'difficulty_{difficulty:.1f}'] = successes / self.num_test_episodes

        return results

    def run_full_generalization_suite(self) -> Dict[str, Any]:
        """
        Run complete generalization test suite.

        Returns:
            Comprehensive generalization results
        """
        suite_results = {}

        for env_name in self.environments:
            print(f"\nTesting generalization on {env_name}...")

            env_results = {}

            # Test 1: Distribution shift
            print("  - Testing distribution shift...")
            env_results['distribution_shift'] = self.test_distribution_shift(env_name)

            # Test 2: Unseen seeds
            print("  - Testing unseen seeds...")
            env_results['unseen_seeds'] = self.test_unseen_seeds(env_name)

            # Test 3: Robustness
            print("  - Testing robustness...")
            env_results['robustness'] = self.test_robustness(env_name)

            suite_results[env_name] = env_results

        return suite_results


class ComputeMetrics:
    """
    Compute various metrics for analysis.
    """

    @staticmethod
    def generalization_gap(
        training_performance: float,
        test_performance: float,
    ) -> float:
        """Compute generalization gap"""
        return training_performance - test_performance

    @staticmethod
    def relative_performance(
        agent_performance: float,
        baseline_performance: float,
    ) -> float:
        """Compute relative performance to baseline"""
        if baseline_performance == 0:
            return 0.0
        return agent_performance / baseline_performance

    @staticmethod
    def transfer_efficiency(
        transfer_performance: float,
        scratch_performance: float,
    ) -> float:
        """
        Compute transfer learning efficiency.
        How much better/worse compared to training from scratch?
        """
        if scratch_performance == 0:
            return 1.0
        return transfer_performance / scratch_performance


if __name__ == "__main__":
    print("Generalization Testing Module")
    print("Use for evaluating agent generalization on unseen environments")
    print("\nExample:")
    print("  tester = GeneralizationTester(agent, environments)")
    print("  results = tester.run_full_generalization_suite()")
