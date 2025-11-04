
# 4 domains (all running on your laptop)

# COMA — Multi-Domain Cross-Game Generalization

A practical research playground for training multi-agent RL policies that generalize across very different games. Built around a COMA-based solver, adversarial level generators, and a meta-RL inference layer so agents can figure out what game they're in and adapt on the fly.

Table of contents
- Overview
- Quick start
- What this repo contains
- The four domains (short)
- Design decisions & research contributions
- Folder layout
- Metrics, tech stack, and tips
- Contributing, license, contact

## Overview

I got tired of "implement algorithm X on environment Y and call it a day." This project asks: can multi-agent coordination skills learned in one type of game transfer to completely different types of games? Can a level generator automatically find that sweet spot between easy and impossible? Can a policy infer the game type from a short history and adapt without explicit task labels?

If that sounds a little ambitious, good — that's the point. The goal is reproducible, laptop-first research that combines: COMA credit assignment, procedural and adversarial level generation, cross-domain transfer, and a small meta-RL inference module.

Light joke (because README authors apparently need a personality): I train agents to deal with unpredictable levels so they can handle my surprises when I forget to save.

## Quick start

These commands assume you have Python 3.8+ and pip installed. The real repo includes detailed requirements; below are the quick steps you can try once the code is present.

Install dependencies:

```bash
pip install -r requirements.txt
```
```bash
python scripts/train_single_domain.py --episodes 10000
```

Evaluate on unseen levels:

```bash
python scripts/test_unseen.py --checkpoint checkpoints/best_model.pt
```

Transfer to an Atari-like domain (fine-tune late layers):

```bash
python scripts/train_transfer.py --pretrained checkpoints/best_model.pt
```

Adversarial training (generator vs solver):

```bash
```

Note: some scripts above are placeholders in the README — they're accurate descriptions of pipeline steps but check the `scripts/` folder for the canonical CLI flags.

## What this repo contains (short)


- environments/: procedural dungeon engine, PettingZoo wrappers for Atari, and a learned level generator.
- agents/: COMA implementation, transformer-based inference policy, centralized critics and helper modules.
- training/: curriculum & adversarial trainers, transfer learner utilities.
- evaluation/: benchmarks, generalization tests, and analysis scripts.
Joke: the repo is like a buffet for RL — take a little bit of everything and don't blame me if you overfit.

## The four domains (summary)

1) Procedural Dungeons
- Grid-based randomized maps with enemies, treasures, and emergent coordination needs. Training here forces policies to learn general strategies rather than memorized routes.

- Fine-tune pretrained coordination features on multi-agent Pong, Tennis, or similar PettingZoo environments. Freezing early layers lets the network reuse coordination priors.

3) Adversarial Level Generation
- A generator agent is trained to produce levels that are hard-but-solvable. Generator and solver co-evolve; it's an automatic curriculum.

- A small Transformer reads a short trajectory buffer (e.g., last 20 transitions) and infers the current game type so the policy can adapt quickly with minimal fine-tuning.

## Research contributions

- Adversarial curriculum learning: generator–solver co-evolution to find the difficulty frontier.
- Cross-domain generalization: demonstrate that coordination skills can transfer across very different games.
- Inference-based adaptation: task inference from trajectories enables zero-shot or fast adaptation without labeled task IDs.

## Folder layout (canonical)
```
multi_domain_coma/
│   ├── procedural_dungeon.py
│   ├── atari_wrapper.py
├── agents/
│   ├── coma_agent.py
│   ├── transformer_policy.py
│   └── critics.py
├── training/
│   ├── curriculum_scheduler.py
│   ├── adversarial_trainer.py
├── evaluation/
│   ├── benchmarks.py
│   └── analysis.py
├── scripts/
│   ├── train_single_domain.py
│   ├── train_transfer.py
│   ├── train_meta_rl.py
│   └── test_unseen.py
├── checkpoints/
└── README.md
```

## Metrics tracked (examples)
- Win rate on unseen procedurally generated levels
- Transfer efficiency (time or episodes to target performance on new domain)
- Curriculum difficulty curve (how generator difficulty evolves)
- Inference accuracy for the Transformer task classifier

## Tech stack
- Core: Python + PyTorch (PyTorch Lightning used in experiments)
- Multi-agent environments: PettingZoo for Atari-style games
- Environment engine: NumPy-based dungeon + optional Pygame visualization
- Logging: TensorBoard + Weights & Biases
- Analysis: Matplotlib, Seaborn

## Practical tips & troubleshooting
- Use a virtual environment (venv or conda) to avoid dependency conflicts.
- If training is slow on CPU, reduce batch sizes, lower environment resolution, or use fewer parallel env workers.
- Reproducibility: fix random seeds in both NumPy and PyTorch for experiments you want to compare.
- Missing GPU: the pipelines are CPU-first by design; enabling GPU requires installing CUDA-enabled PyTorch and may need small changes in Lightning Trainer flags.

Quick troubleshooting checklist:

- "ImportError" for PettingZoo or PyTorch: check `pip install -r requirements.txt` and Python version.
- "CUDA out of memory": lower batch size or switch to CPU mode for debugging.

## Contributing

Contributions are welcome. Good first issues include: cleaning up the scripts' CLI, adding unit tests for the environment dynamics, and improving visualization utilities.


1. Open an issue describing the change.
2. Add tests where applicable.
3. Submit a PR with a concise description and benchmarks if you changed training behavior.

Light policy joke: PRs that fix typos are more likely to be merged than those that only add new hyperparameters.
This repository is intended for research and educational use. Add a LICENSE file to set the exact terms (MIT or similar recommended).

If you build on this work, please include a short citation in your paper or README describing the ideas behind COMA + adversarial curriculum + meta inference.

## Contact

Author: Yuvanesh Sankar
Email: (add your contact email here)

If you'd like, I can also:

- Add a minimal requirements.txt if it's missing.
- Add a short example notebook demonstrating training/evaluation on a tiny environment.

Completion note: README was reformatted for clarity, with short jokes and practical tips added. If you'd like a different tone (more formal or more playful), tell me which sections to adjust.

## Deep dive (developer guide) — file-by-file explanation

This section explains what each important module does, why it exists, and how the pieces connect. Read this if you want to understand the full pipeline from environments -> agents -> trainers -> evaluation.
```
multi_domain_coma/
├── environments/         # All environment code (dungeon, Atari wrappers, level generator)
├── agents/               # Agent implementations (COMA, transformer/meta-RL, critics)
├── training/             # Trainers, schedulers, transfer utilities
├── evaluation/           # Benchmark & generalization tests
├── scripts/              # Top-level training/eval scripts
├── checkpoints/          # Saved models
├── logs/                 # Logs and tensorboard outputs
└── README.md             # (you are here)
```

Note: I ran a quick consistency pass and fixed one import bug (a file that had a stray leading space in its filename). See "Known fixes" below.

Environments (what each file implements and why)

- `environments/procedural_dungeon.py`
	- Implements `ProceduralDungeonEnv`, a multi-agent grid-world with procedurally generated maps.
	- Why: trains agents on diverse layouts so policies learn coordination strategies (not memorized paths).
	- Observations: each agent receives a local grid (flattened), agent health, team health, position and enemies_remaining.
	- Actions: Discrete(6) — [up, down, left, right, attack, stay].
	- Rewards: mixture of individual rewards and team-level bonuses (treasure, clearing enemies). Team reward encourages coordinated behavior.

- `environments/atari_wrapper.py`
	- Wraps PettingZoo multi-agent Atari environments and converts frames to a common observation format.
	- Why: provides the cross-domain target environments (Pong, Tennis, Boxing) for transfer experiments.
	- Key features: frame preprocessing (grayscale, downsample), frame-stacking, `step_list` and `step` interfaces, optional `AtariTransferWrapper` which maps dungeon-style actions to Atari actions for transfer experiments.
	- Notes: this file uses SciPy (`scipy.ndimage.zoom`) for downsampling — include `scipy` in dependencies if you plan to run Atari experiments.

- `environments/level_generator.py`
	- Contains `LevelGenerator` (neural generator) and `AdversarialLevelGenerator` (RL wrapper) plus `GeneratedDungeonEnv`.
	- Why: generator creates procedurally adversarial levels (hard-but-solvable) so the solver agents receive a growing curriculum automatically.
	- `GeneratedDungeonEnv` sits between generator and `ProceduralDungeonEnv` to apply generated positions to a running environment instance.
	- Important integration note: `GeneratedDungeonEnv` accepts either a base environment instance (`base_env`) or a class (`base_env_class`) to be flexible for how scripts construct it.

Agents (what each file implements)

- `agents/coma_agent.py`
	- Full COMA implementation: decentralized `Actor`s and a centralized `Critic` for counterfactual policy gradients.
	- Replay buffer, target critic, actor/critic optimizers, helper methods to convert joint actions to one-hot, greedy-action baselines, counterfactual advantage and policy updates.
	- Dueling option: the actor can be built with a dueling-style decomposition (value + advantage). This is more common in value-based methods, but here it's available as an architecture choice for the policy head.
	- Typical usage: `select_actions`, `store_experience`, `update`, `save` and `load`.

- `agents/transformer_policy.py`
	- Implements the meta-RL policy: transformer-based `TaskInferenceNetwork` + `ConditionalPolicyNetwork`.
	- Why: the transformer reads a short trajectory window (e.g., last 20 transitions) and infers a task embedding that conditions the policy for fast adaptation (zero-shot or few-shot transfer between domains).
	- Exposes `TransformerMetaRLPolicy` with `process_trajectory_step`, `infer_task`, `select_action`, `get_value`, and `update` methods.

- `agents/critics.py`
	- Convenience collection of critic implementations (state critic, action critic, centralized critic, advantage-weighted critic, multi-head critic).
	- These modules are used across the training code and experiments depending on which algorithm (COMA vs QMIX vs independent critics) you're evaluating.

Training and transfer (how the different training files connect)

- `training/curriculum_scheduler.py`
	- Controls how difficulty and environment parameters evolve across training (sigmoid schedule, phased schedule, etc.).

- `training/adversarial_trainer.py`
	- Alternates between solver training and generator updates. Typical cycle: train solver for X episodes on generator-produced levels → evaluate solver performance → use generator to produce new levels and update generator based on solver difficulty.

- `training/transfer_learner.py`
	- Implements progressive neural networks and `TransferLearner` utilities.
	- Strategies supported: `fine_tuning`, `feature_extraction`, and a progressive network approach with lateral connections.
	- What is transferred? Typically: early layers / feature extractors that capture coordination priors (how to behave as a team). The transfer code copies matching parameters from the source actor and allows different strategies for fine-tuning target-domain heads.
	- Important: The script `scripts/train_transfer.py` originally passed an environment object to `TransferLearner` (bug). I updated the script to build a small target model (MLP that flattens Atari frames then outputs logits) and pass that model to `TransferLearner` — this is a pragmatic bridge until a full conv-net feature extractor is implemented.

Scripts (how to run each high-level pipeline)

- `scripts/train_single_domain.py`
	- Trains COMA in the procedural dungeon using the curriculum scheduler. Typical entrypoint for baseline single-domain training.

- `scripts/train_transfer.py`
	- Loads a pre-trained COMA agent from the dungeon and performs transfer training to an Atari-like target.
	- What we transfer: by default the script copies what matches between the source actor and the target model (see `TransferLearner._initialize_target_from_source`). The typical idea is freezing early layers (coordination priors) and only training the top (task-specific) layers.
	- Note: I added a simple target MLP so the `TransferLearner` receives a `nn.Module` rather than an environment object.

- `scripts/train_adversarial.py`
	- Runs generator vs solver co-evolution. Creates `AdversarialLevelGenerator`, a `COMAAgent` solver, wraps the dungeon in `GeneratedDungeonEnv`, and runs adversarial cycles via `AdversarialTrainer`.
	- Integration fix: `GeneratedDungeonEnv` now accepts a base environment instance (this matches how the script constructs it).

- `scripts/train_meta_rl.py`
	- Trains the transformer-based inference policy across multiple procedural-dungeon tasks (different difficulty settings). The transformer produces a task embedding used by the conditional policy to adapt.

Evaluation

- `evaluation/benchmarks.py`, `evaluation/generalization_tests.py`, `evaluation/analysis.py`
	- Helpers for comparing COMA to baselines, running generalization experiments on unseen levels, and plotting/ablation analysis.

Why transfer learning here and what gets transferred

- Motivation
	- Coordination priors (e.g., when to attack, when to defend, how to position relative to teammates) are often domain-agnostic. If an agent learns coordination in procedurally generated dungeons, those early representations (attention to teammates, high-level tactics) can speed up learning on other games that require teamwork (multi-agent Pong, Tennis).

- What we transfer
	- Typically, we transfer early neural network layers (feature extractors) that capture coordination features. The training scripts implement two strategies:
		1. Feature extraction: freeze early layers and only train the top (task-specific) layers on the target domain.
		2. Fine-tuning: initialize target with source weights and train with a (possibly smaller) learning rate.
	- The repository also contains a Progressive Neural Network implementation that incrementally adds a target column with lateral connections to reuse previously learned features while avoiding catastrophic forgetting.

Detailed environment semantics (how observations/actions map)

- Dungeon env (`ProceduralDungeonEnv`)
	- Observation (per agent):
		- 'grid': flattened local grid of size (2*vision_range + 1)^2 (integers encoding walls, agents, enemies, treasures)
		- 'agent_health': scalar
		- 'team_health': vector of all agents' health
		- 'position': 2D position
		- 'enemies_remaining': scalar
	- Action space: Discrete(6) mapping to movements and attack/stay.
	- Reward: small step penalty, attack/treasure rewards, team bonus for clearing enemies.

- Atari wrapper (`AtariMultiAgentWrapper` / `AtariTransferWrapper`)
	- Observation: stacked frames (frame_stack, H, W, C) + small metadata (agent_id, scores).
	- For transfer experiments the wrapper includes `map_dungeon_action` to map dungeon discrete actions into Atari actions.

Quick sanity checks and how I validated integrations

- I ran a static repository scan and exercised the major modules by loading them (not full training):
	- Ensured `environments.level_generator` is importable (fixed a filename issue where the file had an accidental leading space).
	- Fixed `scripts/train_transfer.py` so `TransferLearner` receives a `nn.Module` target model, not an environment instance.
	- Adjusted `GeneratedDungeonEnv` to accept either a live environment instance or a class; scripts that pass instances will now work.

Known issues & recommendations

- External dependencies: several modules rely on optional packages (PettingZoo, SciPy, Gymnasium, Pygame). Add these to `requirements.txt` when you run experiments. I can create a minimal `requirements.txt` for you — tell me if you want that.
- Performance: Atari experiments require a conv-backbone to benefit from transfer; the placeholder MLP in `train_transfer.py` is a simple stopgap. For serious transfer, add a CNN encoder for frames and align the source encoder architecture accordingly.
- Unit tests: there are small example usage blocks in the modules; adding a tiny test suite (pytest) that imports each module and runs the example code would help catch runtime import issues earlier.

Minimal recommended `requirements.txt` (suggested)

```
torch
numpy
gymnasium
pettingzoo
scipy
pytorch-lightning
matplotlib
tensorboard
wandb
```

If you want, I can add the `requirements.txt` now.

What I changed in code just now

- Renamed/fixed a misnamed file (removed a leading-space filename) and added `environments/level_generator.py`.
- Appended the adversarial generator and `GeneratedDungeonEnv` to `environments/level_generator.py` and made `GeneratedDungeonEnv` accept an env instance.
- Updated `scripts/train_transfer.py` to create a simple `target_model` (MLP) and pass it to `TransferLearner` instead of passing an environment object.

Next steps I can do for you (pick one or more):

1. Auto-generate a `requirements.txt` with pinned versions and add it to the repo.
2. Create a short example notebook (or script) that runs a 1-episode smoke test for `train_single_domain.py` with tiny sizes to validate end-to-end behavior.
3. Add a small test suite (pytest) that imports each module and runs the embedded example usage blocks.
4. Rewrite the README tone (more formal / research-paper style) or add diagrams (text-based ASCII or link to images).

Tell me which next step you want and I will implement it. If you prefer I continue and fully rewrite README into a longer developer manual, say "Please expand README fully" and I will do that next.

