

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum


class TrainingPhase(Enum):
    """Training phases for curriculum learning"""
    DOMAIN_1_EASY = 1
    DOMAIN_1_MEDIUM = 2
    DOMAIN_1_HARD = 3
    DOMAIN_2_TRANSFER = 4
    ADVERSARIAL_GEN = 5
    META_RL = 6


class CurriculumScheduler:
    """
    Manages curriculum learning progression.
    Automatically increases difficulty and transitions between training phases.
    """

    def __init__(
        self,
        total_episodes: int,
        num_phases: int = 6,
        initial_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
        difficulty_schedule: str = 'sigmoid',  # 'linear', 'sigmoid', 'exponential'
    ):
        """
        Initialize curriculum scheduler.

        Args:
            total_episodes: Total training episodes
            num_phases: Number of training phases
            initial_difficulty: Starting difficulty
            max_difficulty: Maximum difficulty
            difficulty_schedule: Type of difficulty progression
        """
        self.total_episodes = total_episodes
        self.num_phases = num_phases
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_schedule = difficulty_schedule

        self.current_episode = 0
        self.current_phase = TrainingPhase.DOMAIN_1_EASY
        self.episodes_per_phase = total_episodes // num_phases

        # Phase transitions
        self.phase_boundaries = self._compute_phase_boundaries()

        # Difficulty tracking
        self.difficulty_history = []
        self.phase_history = []

    def _compute_phase_boundaries(self) -> Dict[TrainingPhase, Tuple[int, int]]:
        """Compute episode ranges for each phase"""
        boundaries = {}
        phases = list(TrainingPhase)

        for i, phase in enumerate(phases):
            start = i * self.episodes_per_phase
            end = (i + 1) * self.episodes_per_phase if i < len(phases) - 1 else self.total_episodes
            boundaries[phase] = (start, end)

        return boundaries

    def _compute_difficulty(self, progress: float) -> float:
        """
        Compute difficulty based on training progress.

        Args:
            progress: Progress ratio [0, 1]

        Returns:
            Difficulty in [initial_difficulty, max_difficulty]
        """
        if self.difficulty_schedule == 'linear':
            # Linear progression
            difficulty = self.initial_difficulty + (self.max_difficulty - self.initial_difficulty) * progress

        elif self.difficulty_schedule == 'sigmoid':
            # Sigmoid: slow start, rapid middle, slow end
            difficulty = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))
            difficulty = self.initial_difficulty + (self.max_difficulty - self.initial_difficulty) * difficulty

        elif self.difficulty_schedule == 'exponential':
            # Exponential: rapid early, slow late
            difficulty = self.initial_difficulty + (self.max_difficulty - self.initial_difficulty) * (progress ** 1.5)

        else:
            difficulty = self.initial_difficulty

        return np.clip(difficulty, self.initial_difficulty, self.max_difficulty)

    def step(self) -> Dict[str, float]:
        """
        Advance scheduler by one episode.

        Returns:
            Dictionary with current difficulty and phase info
        """
        self.current_episode += 1

        # Compute overall progress
        overall_progress = self.current_episode / self.total_episodes

        # Compute difficulty
        difficulty = self._compute_difficulty(overall_progress)
        self.difficulty_history.append(difficulty)

        # Update phase
        for phase, (start, end) in self.phase_boundaries.items():
            if start <= self.current_episode < end:
                self.current_phase = phase
                break

        self.phase_history.append(self.current_phase.value)

        # Compute phase-specific difficulty (within-phase progression)
        phase_start, phase_end = self.phase_boundaries[self.current_phase]
        phase_progress = (self.current_episode - phase_start) / (phase_end - phase_start)

        return {
            'episode': self.current_episode,
            'overall_progress': overall_progress,
            'difficulty': difficulty,
            'phase': self.current_phase.value,
            'phase_name': self.current_phase.name,
            'phase_progress': phase_progress,
        }

    def get_current_difficulty(self) -> float:
        """Get current difficulty level"""
        return self.difficulty_history[-1] if self.difficulty_history else self.initial_difficulty

    def get_current_phase(self) -> TrainingPhase:
        """Get current training phase"""
        return self.current_phase

    def should_transition_phase(self, performance_metric: float, threshold: float = 0.7) -> bool:
        """
        Check if should transition to next phase based on performance.

        Args:
            performance_metric: Current performance (e.g., win rate)
            threshold: Performance threshold for transition

        Returns:
            Whether to transition
        """
        # Transition if performance exceeds threshold
        return performance_metric >= threshold

    def get_schedule_info(self) -> Dict:
        """Get full schedule information"""
        return {
            'total_episodes': self.total_episodes,
            'current_episode': self.current_episode,
            'num_phases': self.num_phases,
            'current_phase': self.current_phase.name,
            'episodes_per_phase': self.episodes_per_phase,
            'phase_boundaries': {
                phase.name: boundaries
                for phase, boundaries in self.phase_boundaries.items()
            },
        }


class DifficultyAdapterOnline:
    """
    Online difficulty adaptation based on agent performance.
    Adjusts difficulty in real-time during training.
    """

    def __init__(
        self,
        initial_difficulty: float = 0.3,
        min_difficulty: float = 0.1,
        max_difficulty: float = 0.9,
        window_size: int = 100,
        target_win_rate: float = 0.6,
    ):
        """
        Initialize online difficulty adapter.

        Args:
            initial_difficulty: Starting difficulty
            min_difficulty: Minimum difficulty
            max_difficulty: Maximum difficulty
            window_size: Number of recent episodes for averaging
            target_win_rate: Target success rate for stability
        """
        self.difficulty = initial_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.window_size = window_size
        self.target_win_rate = target_win_rate

        self.win_history = []
        self.difficulty_history = []

    def update(self, success: bool) -> float:
        """
        Update difficulty based on success/failure.

        Args:
            success: Whether episode was successful

        Returns:
            Updated difficulty
        """
        self.win_history.append(float(success))

        # Keep only recent window
        if len(self.win_history) > self.window_size:
            self.win_history.pop(0)

        # Compute recent win rate
        recent_win_rate = np.mean(self.win_history)

        # Adapt difficulty
        if recent_win_rate > self.target_win_rate + 0.1:
            # Too easy, increase difficulty
            self.difficulty = min(self.difficulty * 1.05, self.max_difficulty)
        elif recent_win_rate < self.target_win_rate - 0.1:
            # Too hard, decrease difficulty
            self.difficulty = max(self.difficulty * 0.95, self.min_difficulty)

        self.difficulty_history.append(self.difficulty)

        return self.difficulty

    def get_statistics(self) -> Dict[str, float]:
        """Get difficulty adaptation statistics"""
        return {
            'current_difficulty': self.difficulty,
            'recent_win_rate': np.mean(self.win_history) if self.win_history else 0.0,
            'avg_difficulty': np.mean(self.difficulty_history) if self.difficulty_history else self.difficulty,
        }


class PhaseScheduler:
    """
    Explicit phase scheduling without difficulty changes.
    Useful for structured training with distinct phases.
    """

    def __init__(self, phase_configs: List[Dict]):
        """
        Initialize phase scheduler.

        Args:
            phase_configs: List of phase configurations
                Each config should have: 'name', 'duration', 'config' (dict of params)
        """
        self.phase_configs = phase_configs
        self.current_phase_idx = 0
        self.episodes_in_phase = 0
        self.total_episodes = sum(cfg['duration'] for cfg in phase_configs)

    def step(self) -> Dict:
        """
        Advance to next episode and return current phase config.

        Returns:
            Current phase configuration
        """
        current_config = self.phase_configs[self.current_phase_idx]

        self.episodes_in_phase += 1

        # Check if should transition to next phase
        if self.episodes_in_phase >= current_config['duration']:
            self.current_phase_idx = min(self.current_phase_idx + 1, len(self.phase_configs) - 1)
            self.episodes_in_phase = 0

        return {
            'phase_idx': self.current_phase_idx,
            'phase_name': current_config['name'],
            'episodes_in_phase': self.episodes_in_phase,
            'phase_duration': current_config['duration'],
            'config': current_config.get('config', {}),
        }

    def get_current_phase(self) -> Dict:
        """Get current phase configuration"""
        current = self.phase_configs[self.current_phase_idx]
        return {
            'name': current['name'],
            'config': current.get('config', {}),
        }


# Example curriculum configurations

SINGLE_DOMAIN_CURRICULUM = [
    {
        'name': 'Domain1_Easy',
        'duration': 2000,
        'config': {'num_agents': 2, 'grid_size': 8, 'difficulty': 0.2}
    },
    {
        'name': 'Domain1_Medium',
        'duration': 3000,
        'config': {'num_agents': 2, 'grid_size': 10, 'difficulty': 0.5}
    },
    {
        'name': 'Domain1_Hard',
        'duration': 3000,
        'config': {'num_agents': 2, 'grid_size': 12, 'difficulty': 0.8}
    },
]

MULTI_DOMAIN_CURRICULUM = [
    {
        'name': 'Domain1_Easy',
        'duration': 2000,
        'config': {'domain': 'dungeon', 'difficulty': 0.2}
    },
    {
        'name': 'Domain1_Medium',
        'duration': 2500,
        'config': {'domain': 'dungeon', 'difficulty': 0.5}
    },
    {
        'name': 'Domain1_Hard',
        'duration': 2500,
        'config': {'domain': 'dungeon', 'difficulty': 0.8}
    },
    {
        'name': 'Atari_Transfer',
        'duration': 3000,
        'config': {'domain': 'atari', 'game': 'pong', 'pretrained': True}
    },
    {
        'name': 'AdversarialGen',
        'duration': 3000,
        'config': {'domain': 'dungeon', 'use_generator': True}
    },
    {
        'name': 'MetaRL',
        'duration': 3000,
        'config': {'domain': 'mixed', 'use_task_inference': True}
    },
]


if __name__ == "__main__":
    print("Testing CurriculumScheduler...")

    # Test basic curriculum
    scheduler = CurriculumScheduler(
        total_episodes=10000,
        num_phases=6,
        difficulty_schedule='sigmoid'
    )

    # Simulate training
    for episode in range(100):
        info = scheduler.step()
        if episode % 20 == 0:
            print(f"Episode {info['episode']}: "
                  f"Difficulty {info['difficulty']:.3f}, "
                  f"Phase {info['phase_name']}")

    print("\nSchedule Info:")
    print(scheduler.get_schedule_info())

    # Test online adaptation
    print("\n\nTesting DifficultyAdapterOnline...")
    adapter = DifficultyAdapterOnline(target_win_rate=0.6)

    for i in range(50):
        success = np.random.random() > 0.5
        new_difficulty = adapter.update(success)

        if i % 10 == 0:
            print(f"Step {i}: Success={success}, New Difficulty={new_difficulty:.3f}")

    print("\nAdapter Stats:")
    print(adapter.get_statistics())

    # Test phase scheduler
    print("\n\nTesting PhaseScheduler...")
    phase_scheduler = PhaseScheduler(MULTI_DOMAIN_CURRICULUM)

    for step in range(5):
        phase_info = phase_scheduler.step()
        print(f"Step {step}: {phase_info['phase_name']}, Config: {phase_info['config']}")

    print("\nAll tests passed!")
