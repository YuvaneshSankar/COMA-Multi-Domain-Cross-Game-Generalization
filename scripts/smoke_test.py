"""
Quick smoke test for repository modules.
This script imports core modules and runs tiny example calls.
It is written to be robust if optional dependencies (PettingZoo) are missing.
"""
import traceback
import numpy as np

def try_run(fn, name):
    print(f"\n==> Running: {name}")
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[ERR] {name}: {e}")
        traceback.print_exc()


def test_procedural_dungeon():
    from environments.procedural_dungeon import ProceduralDungeonEnv

    env = ProceduralDungeonEnv(num_agents=2, grid_size=8, num_enemies=1, num_treasures=1, num_obstacles=2)
    obs, info = env.reset()
    assert isinstance(obs, list) and len(obs) == 2
    actions = [0, 5]
    obs2, rewards, terminated, truncated, info = env.step(actions)
    print("rewards:", rewards)


def test_level_generator():
    from environments.level_generator import LevelGenerator, AdversarialLevelGenerator

    gen = AdversarialLevelGenerator(grid_size=8, latent_dim=16)
    level = gen.generate_level(0.5)
    print("level keys:", list(level.keys()))


def test_coma_agent():
    from agents.coma_agent import COMAAgent

    agent = COMAAgent(num_agents=2, state_dim=8, action_dim=6, device='cpu')
    # create fake obs: one per agent
    obs = [np.zeros(8), np.zeros(8)]
    actions = agent.select_actions(obs, exploration=False)
    print("actions:", actions)


def test_transformer_policy():
    from agents.transformer_policy import TransformerMetaRLPolicy

    policy = TransformerMetaRLPolicy(obs_dim=8, action_dim=6, task_latent_dim=16, trajectory_window=5, device='cpu')
    # simulate small trajectory
    for _ in range(6):
        policy.process_trajectory_step(np.zeros(8), np.random.randint(6), 0.0)
    action = policy.select_action(np.zeros(8))
    print("meta action:", action)


def test_atari_wrapper():
    try:
        from environments.atari_wrapper import make_atari_env
    except Exception:
        print("PettingZoo or related deps not installed; skipping Atari wrapper test")
        return

    env = make_atari_env('pong', obs_size=32, frame_stack=2)
    obs, info = env.reset()
    print("Atari env created. num_agents:", env.num_agents)


if __name__ == '__main__':
    print("Running smoke tests... (this should be fast and tolerant of missing optional deps)")

    tests = [
        (test_procedural_dungeon, 'ProceduralDungeonEnv'),
        (test_level_generator, 'LevelGenerator'),
        (test_coma_agent, 'COMAAgent'),
        (test_transformer_policy, 'TransformerMetaRLPolicy'),
        (test_atari_wrapper, 'AtariWrapper (optional)'),
    ]

    for fn, name in tests:
        try_run(fn, name)

    print('\nSmoke tests finished.')
