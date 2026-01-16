# MOHITO Baseline Training Guide

This document describes how to train MOHITO models for the rideshare domain (upstream) and use them for zero-shot evaluation on our microtransit system.

## Quick Start

```powershell
cd baselines/mohito-public/rideshare
python -m mohitoR.trainer
```

Training outputs are saved to `results/<experiment_name>/`.

---

## Prerequisites

1. **Python 3.8+** with PyTorch and torch_geometric
2. Install MOHITO dependencies:
   ```powershell
   cd baselines/mohito-public/rideshare
   pip install -e .
   ```

---

## Training Parameters

Edit `mohitoR/trainer.py` lines 21-75 to configure:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_episodes` | 100 | Total training episodes |
| `num_agents` | 4 | Number of concurrent agents |
| `lr_actor` | 0.001 | Actor learning rate |
| `lr_critic` | 0.01 | Critic learning rate |
| `pool_limit` | 2 | Max passengers per ride |
| `grid_len/grid_wid` | 10 | Grid dimensions |
| `steps_per_episode_train` | 100 | Steps per episode |
| `save_model_every_eps` | 100 | Checkpoint frequency |

### Recommended Configurations

**Quick test (fast):**
```python
num_episodes = 100
num_agents = 2
steps_per_episode_train = 50
```

**Full training (paper settings):**
```python
num_episodes = 20000
num_agents = 4
steps_per_episode_train = 100
```

---

## Output Structure

```
results/cuda_<date>_<config>/
├── model_files/
│   ├── <episode>/
│   │   ├── policy_agent0.pth   ← Actor weights
│   │   ├── policy_agent1.pth
│   │   └── ...
│   └── critic.pth
├── training_stats.csv
└── rewards.pkl
```

---

## Using Trained Models

Copy the best checkpoint to use with our evaluator:

```powershell
# Evaluate with MOHITO
python scripts/run_eval.py --config configs/manhattan.yaml `
    --policy mohito `
    --model-path baselines/mohito-public/rideshare/results/<run>/model_files/<best_ep>/policy_agent0.pth
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: rideshare` | Run `pip install -e .` in rideshare dir |
| CUDA OOM | Reduce `num_agents` or `batch_size` |
| Training stuck | Check `training_stats.csv` for loss values |

---

## Citation

```bibtex
@inproceedings{mohito,
    author = {Anil, Gayathri and Doshi, Prashant and Redder, Daniel and Eck, Adam and Soh, Leen-Kiat},
    booktitle = {UAI},
    title = {MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for Task-Open Systems},
    year = {2025}
}
```
