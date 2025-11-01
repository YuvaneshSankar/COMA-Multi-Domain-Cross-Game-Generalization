
4 domains (all running on your laptop)
Domain 1: Procedurally Generated Dungeons
i built a custom grid-based environment where agents navigate randomized mazes, fight enemies, collect treasures. the cool part? every episode has a different map. no two games are alike. this forces agents to learn generalizable coordination strategies instead of memorizing one layout.

What happens here:

Agents must coordinate to defeat enemies

Centralized critic (COMA) decides if agent A's action was actually good given what agent B did

Agents learn: "should i attack or should i defend? depends on what my teammate is doing"

Domain 2: Cross-Game Transfer (Atari)
once agents get decent at coordination in dungeons, i throw them into completely different games—multi-agent Pong, Tennis, Boxing (using PettingZoo).

the transfer trick:

i freeze the early neural network layers (those learned "how to coordinate in general")

only fine-tune the later layers (those learn "this specific game has different physics")

result: agents learn the new game ~30-50% faster than if they started from scratch

why? because coordination skills are transferable. attacking together in a dungeon feels kinda similar to attacking together in Pong, even though the games look nothing alike.

Domain 3: Adversarial Level Generation
here's where it gets fun. i decided to train another agent whose job is to generate game levels for the COMA team to solve.

this generator agent:

wins when it creates levels that are "hard but not impossible"

loses if levels are too easy (agents solve instantly) or impossible (unsolvable)

continuously evolves to find the sweet spot

what this means:

the solver agents get progressively harder challenges without me manually tweaking difficulty

the generator learns what makes a level "hard" for multi-agent teams

both agents co-evolve: generator creates harder levels → solvers adapt → generator creates even harder levels

it's like a curriculum that designs itself.

Domain 4: Meta-RL with Game-Type Inference
this is the kicker. i wanted to train one policy that works across all these different domains without telling it which domain it's in.

how?

the agent keeps a memory of the last 20 transitions (states, actions, rewards)

a Transformer network reads this memory and figures out: "wait, based on what i'm seeing... am i in a dungeon? Pong? a generated level?"

once it infers the game type, it automatically switches its strategy

why this matters:

zero-shot adaptation: throw the agent into a brand new game it's never seen, and it should still perform decently because it's learned how different games "feel"

this is basically how humans play games. we play Mario, then start playing Zelda, and we don't need a manual to understand we should attack enemies differently

the research contributions (why this isn't just "cool" but actually publishable)
1. novel combined approach
nobody's doing COMA + procedural generation + transfer learning + meta-RL all in one system. i'm combining insights from 4 different subfields of RL.

2. adversarial curriculum learning
the generator-solver co-evolution is genuinely novel. instead of having a human manually design difficulty curves, the system learns to create its own.

3. cross-domain generalization
showing that multi-agent coordination skills transfer across fundamentally different game types is a concrete research contribution.

4. inference-based adaptation
meta-RL with implicit game-type inference (no labels) is harder than explicit curriculum learning, and it's more realistic.

folder structure (so you know what's where)
text
multi_domain_coma/
│
├── environments/
│   ├── procedural_dungeon.py      # my custom grid-based game engine
│   ├── atari_wrapper.py            # wraps PettingZoo Atari games
│   └── level_generator.py          # the agent that generates levels
│
├── agents/
│   ├── coma_agent.py               # the core COMA algorithm i implemented
│   ├── transformer_policy.py       # Transformer for meta-RL inference

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

Train a single-domain solver (procedural dungeon):

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
python scripts/train_adversarial.py --difficulty-range 0.3-0.9
```

Note: some scripts above are placeholders in the README — they're accurate descriptions of pipeline steps but check the `scripts/` folder for the canonical CLI flags.

## What this repo contains (short)

High level: environments, agents, training loops, evaluation scripts, and utilities for logging and visualization.

- environments/: procedural dungeon engine, PettingZoo wrappers for Atari, and a learned level generator.
- agents/: COMA implementation, transformer-based inference policy, centralized critics and helper modules.
- training/: curriculum & adversarial trainers, transfer learner utilities.
- evaluation/: benchmarks, generalization tests, and analysis scripts.

Joke: the repo is like a buffet for RL — take a little bit of everything and don't blame me if you overfit.

## The four domains (summary)

1) Procedural Dungeons
- Grid-based randomized maps with enemies, treasures, and emergent coordination needs. Training here forces policies to learn general strategies rather than memorized routes.

2) Cross-Game Transfer (Atari-style)
- Fine-tune pretrained coordination features on multi-agent Pong, Tennis, or similar PettingZoo environments. Freezing early layers lets the network reuse coordination priors.

3) Adversarial Level Generation
- A generator agent is trained to produce levels that are hard-but-solvable. Generator and solver co-evolve; it's an automatic curriculum.

4) Meta-RL with Inference
- A small Transformer reads a short trajectory buffer (e.g., last 20 transitions) and infers the current game type so the policy can adapt quickly with minimal fine-tuning.

## Research contributions

- Novel combined system: COMA + procedural generation + adversarial curriculum + meta-RL in one pipeline.
- Adversarial curriculum learning: generator–solver co-evolution to find the difficulty frontier.
- Cross-domain generalization: demonstrate that coordination skills can transfer across very different games.
- Inference-based adaptation: task inference from trajectories enables zero-shot or fast adaptation without labeled task IDs.

## Folder layout (canonical)

```
multi_domain_coma/
├── environments/
│   ├── procedural_dungeon.py
│   ├── atari_wrapper.py
	└── level_generator.py
├── agents/
│   ├── coma_agent.py
│   ├── transformer_policy.py
│   └── critics.py
├── training/
│   ├── curriculum_scheduler.py
│   ├── adversarial_trainer.py
│   └── transfer_learner.py
├── evaluation/
│   ├── benchmarks.py
│   ├── generalization_tests.py
│   └── analysis.py
├── scripts/
│   ├── train_single_domain.py
│   ├── train_transfer.py
│   ├── train_adversarial.py
│   ├── train_meta_rl.py
│   └── test_unseen.py
├── checkpoints/
├── logs/
└── README.md
```

## Metrics tracked (examples)

- Win rate on unseen procedurally generated levels
- Transfer efficiency (time or episodes to target performance on new domain)
- Curriculum difficulty curve (how generator difficulty evolves)
- Inference accuracy for the Transformer task classifier
- Generalization gap: seen vs unseen distributions

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
- "Evaluation mismatch vs training": ensure you're using evaluation seeds different from training seeds; save and load checkpoints with strict=False only if network shape changed.

## Contributing

Contributions are welcome. Good first issues include: cleaning up the scripts' CLI, adding unit tests for the environment dynamics, and improving visualization utilities.

If you contribute, please:

1. Open an issue describing the change.
2. Add tests where applicable.
3. Submit a PR with a concise description and benchmarks if you changed training behavior.

Light policy joke: PRs that fix typos are more likely to be merged than those that only add new hyperparameters.

## License & citation

This repository is intended for research and educational use. Add a LICENSE file to set the exact terms (MIT or similar recommended).

If you build on this work, please include a short citation in your paper or README describing the ideas behind COMA + adversarial curriculum + meta inference.

## Contact

Author: Yuvanesh Sankar
Email: (add your contact email here)

If you'd like, I can also:

- Add a minimal requirements.txt if it's missing.
- Add a short example notebook demonstrating training/evaluation on a tiny environment.

Completion note: README was reformatted for clarity, with short jokes and practical tips added. If you'd like a different tone (more formal or more playful), tell me which sections to adjust.

