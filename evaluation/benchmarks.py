"""
Benchmarking Module: Compare COMA with Baseline Algorithms
Evaluates performance across different RL algorithms and domains.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
from datetime import datetime
from collections import defaultdict


class AlgorithmBenchmark:
    """
    Benchmark different RL algorithms on the same tasks.
    Provides standardized evaluation protocol.
    """

    def __init__(
        self,
        algorithms: Dict[str, Any],  # name -> agent instance
        environments: Dict[str, Any],  # name -> env instance
        num_episodes: int = 1000,
        max_steps_per_episode: int = 200,
        num_seeds: int = 3,
    ):
        """
        Initialize benchmarking suite.

        Args:
            algorithms: Dictionary mapping algorithm names to agent instances
            environments: Dictionary mapping environment names to env instances
            num_episodes: Episodes to train per run
            max_steps_per_episode: Max steps per episode
            num_seeds: Number of random seeds for multiple runs
        """
        self.algorithms = algorithms
        self.environments = environments
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.num_seeds = num_seeds

        # Results storage
        self.results = defaultdict(lambda: defaultdict(list))
        self.timing = {}

    def benchmark_all(self) -> Dict[str, Dict[str, List]]:
        """
        Run benchmarks on all algorithm-environment combinations.

        Returns:
            Nested dictionary: algorithm -> environment -> metrics
        """
        total_combinations = len(self.algorithms) * len(self.environments) * self.num_seeds
        current = 0

        for algo_name, agent in self.algorithms.items():
            for env_name, env in self.environments.items():
                for seed in range(self.num_seeds):
                    current += 1
                    print(f"[{current}/{total_combinations}] Benchmarking {algo_name} on {env_name} (seed {seed})")

                    # Run benchmark
                    metrics = self._benchmark_single(algo_name, agent, env, seed)

                    # Store results
                    for metric_name, metric_value in metrics.items():
                        self.results[algo_name][f"{env_name}_{metric_name}"].append(metric_value)

        return dict(self.results)

    def _benchmark_single(
        self,
        algo_name: str,
        agent: Any,
        env: Any,
        seed: int,
    ) -> Dict[str, float]:
        """
        Run single benchmark: one algorithm on one environment.

        Returns:
            Metrics for this run
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Reset environment
        obs, info = env.reset(seed=seed)

        # Tracking
        episode_rewards = []
        episode_lengths = []
        success_count = 0

        # Training loop
        for episode in range(self.num_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            success = False

            for step in range(self.max_steps_per_episode):
                # Select actions
                if hasattr(agent, 'select_actions'):
                    # Multi-agent
                    actions = agent.select_actions(obs, exploration=True)
                else:
                    # Single agent
                    actions = agent.select_action(obs, exploration=True)

                # Step environment
                obs, rewards, terminated, truncated, info = env.step(actions)

                # Accumulate reward
                if isinstance(rewards, list):
                    episode_reward += np.mean(rewards)
                else:
                    episode_reward += rewards

                episode_length += 1

                if terminated or truncated:
                    success = info.get('success', False)
                    break

            # Update agent
            if hasattr(agent, 'update'):
                agent.update(batch_size=32)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if success:
                success_count += 1

        # Compute metrics
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'success_rate': success_count / self.num_episodes,
            'mean_episode_length': np.mean(episode_lengths),
        }

    def compare_results(self) -> Dict[str, Dict]:
        """
        Compare and summarize results across algorithms.

        Returns:
            Summary statistics
        """
        summary = {}

        for algo_name in self.algorithms:
            algo_results = {}

            for env_name in self.environments:
                metrics_data = {}

                # Collect all metrics for this algo-env pair
                for metric_key in self.results[algo_name]:
                    if env_name in metric_key:
                        metric_name = metric_key.replace(f"{env_name}_", "")
                        values = self.results[algo_name][metric_key]

                        metrics_data[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                        }

                algo_results[env_name] = metrics_data

            summary[algo_name] = algo_results

        return summary

    def save_results(self, path: str):
        """Save benchmark results to JSON"""
        summary = self.compare_results()

        # Convert to JSON-serializable format
        json_results = {}
        for algo in summary:
            json_results[algo] = {}
            for env in summary[algo]:
                json_results[algo][env] = {}
                for metric in summary[algo][env]:
                    json_results[algo][env][metric] = {
                        k: float(v) for k, v in summary[algo][env][metric].items()
                    }

        with open(path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {path}")


class PerformanceMetrics:
    """
    Compute detailed performance metrics.
    """

    @staticmethod
    def compute_sample_efficiency(
        training_returns: List[float],
        target_performance: float
    ) -> int:
        """
        Compute number of steps to reach target performance.

        Args:
            training_returns: List of returns over training
            target_performance: Target return value

        Returns:
            Number of steps to reach target
        """
        for i, return_val in enumerate(training_returns):
            if return_val >= target_performance:
                return i

        return len(training_returns)  # Never reached

    @staticmethod
    def compute_convergence_rate(training_returns: List[float]) -> float:
        """
        Compute how quickly algorithm converges.
        Uses slope of moving average.

        Args:
            training_returns: List of returns over training

        Returns:
            Convergence rate (higher = faster)
        """
        if len(training_returns) < 2:
            return 0.0

        # Compute moving average (window=50)
        window = min(50, len(training_returns) // 4)
        moving_avg = np.convolve(training_returns, np.ones(window)/window, mode='valid')

        if len(moving_avg) < 2:
            return 0.0

        # Compute average slope
        diffs = np.diff(moving_avg)
        convergence_rate = np.mean(diffs)

        return convergence_rate

    @staticmethod
    def compute_stability(training_returns: List[float]) -> float:
        """
        Compute training stability (lower variance = more stable).

        Args:
            training_returns: List of returns over training

        Returns:
            Stability score (0-1, higher = more stable)
        """
        if len(training_returns) < 2:
            return 0.0

        # Split training into windows
        window_size = max(1, len(training_returns) // 10)
        window_stds = []

        for i in range(0, len(training_returns) - window_size, window_size):
            window = training_returns[i:i+window_size]
            window_stds.append(np.std(window))

        # Stability = 1 / (1 + avg_std)
        avg_std = np.mean(window_stds) if window_stds else 0.0
        stability = 1.0 / (1.0 + avg_std)

        return stability

    @staticmethod
    def compute_wall_clock_efficiency(
        training_time_seconds: float,
        final_performance: float,
    ) -> float:
        """
        Compute wall-clock efficiency (performance per second).

        Args:
            training_time_seconds: Total training time
            final_performance: Final performance achieved

        Returns:
            Efficiency metric
        """
        if training_time_seconds <= 0:
            return 0.0

        return final_performance / training_time_seconds


class VisualizationHelper:
    """
    Helper for creating benchmark visualizations.
    """

    @staticmethod
    def plot_training_curves(
        results: Dict[str, List[float]],
        save_path: Optional[str] = None,
    ):
        """
        Plot training curves for multiple algorithms.

        Args:
            results: Dict mapping algorithm names to training returns
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        plt.figure(figsize=(12, 6))

        for algo_name, returns in results.items():
            plt.plot(returns, label=algo_name, alpha=0.7)

        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Training Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_box_comparison(
        results: Dict[str, Dict[str, List[float]]],
        metric: str = 'success_rate',
        save_path: Optional[str] = None,
    ):
        """
        Create box plot comparing algorithms on specific metric.

        Args:
            results: Dict: algorithm -> environment -> metrics
            metric: Metric to plot
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(figsize=(12, 6))

        algo_names = list(results.keys())
        metric_values = []

        for algo in algo_names:
            values = []
            for env_metrics in results[algo].values():
                if metric in env_metrics:
                    values.extend(env_metrics[metric])
            metric_values.append(values)

        axes.boxplot(metric_values, labels=algo_names)
        axes.set_ylabel(metric)
        axes.set_title(f'Algorithm Comparison: {metric}')

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    print("Benchmarking Module")
    print("Use for comparing algorithms across environments")
    print("\nExample:")
    print("  benchmark = AlgorithmBenchmark(algorithms, environments)")
    print("  results = benchmark.benchmark_all()")
    print("  benchmark.save_results('results.json')")
