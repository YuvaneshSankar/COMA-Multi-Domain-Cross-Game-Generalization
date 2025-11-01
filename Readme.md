
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

