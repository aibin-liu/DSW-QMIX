# DSW-QMIX

PyTorch code for **constrained multi-agent RL** on the **blocker game** and **vehicular network** benchmarks. The DSW setup uses recurrent agents with dual mixers (reward and cost), joint constrained action selection, and optional monotonicity regularisation (`learners/dsw_learner.py`). Unconstrained baselines **IQL**, **VDN**, and **QMIX** use flat team reward only.

This repository extends the **CMIX** line of work (peak and average constraints); see Citation below.

## Requirements

```bash
pip install -r requirements.txt
# PyTorch, NumPy, SciPy; add matplotlib for plotting scripts.
```

## Training

### Blocker game — DSW (full)

From the repo root:

```bash
./run_blockergame.sh
```

The script enables **soft Polyak targets** and **Huber TD loss** by default (to reduce periodic loss spikes from hard target syncs). Overrides:

| Environment variable | Meaning |
|---------------------|---------|
| `USE_SOFT_TARGET=0` | Disable soft targets; use hard sync every `TARGET_UPDATE_INTERVAL` (default 200). |
| `SOFT_TARGET_TAU` | Polyak coefficient (default `0.005`). |
| `TD_LOSS` | `huber` or `mse` (default `huber`). |
| `HUBER_DELTA` | Huber linear threshold (default `1.0`). |
| `TARGET_UPDATE_INTERVAL` | Hard-target period when soft targets are off. |

Example:

```bash
USE_SOFT_TARGET=0 TARGET_UPDATE_INTERVAL=1000 ./run_blockergame.sh
```

### Blocker game — baselines (IQL, VDN, QMIX)

```bash
./train_blockergame_baselines.sh
```

Optional: `BASELINE_EPOCHS`, `BASELINE_BATCH`, `BASELINE_BUFFER`, `BASELINE_MAX_ENV_T`.

### Vehicular network

```bash
./run_vn.sh
```

Baselines (original workflow):

```bash
python3 run_vn_baselines.py log/vn_baselines [path/to/env.pickle]
```

## Logs and checkpoints

Each run writes under `./log/<log-dir>/`, for example:

- `loss.log`, `global_reward.log`
- For `--rl-model rnn`: `tderror_loss.log`, `mono_loss.log`
- When the environment provides them: `cost.log`, `return.log`, `peak_violation.log`, …
- `env.pickle`, `model/` for checkpoints

## Plotting (blockergame)

```bash
python3 scripts/plot_blockergame_comparison.py --output figures/metrics.png
```

Plots **raw** logs only (rolling mean ± std band and optional Gaussian smoothing). See `python3 scripts/plot_blockergame_comparison.py --help` for `--rolling-window`, `--line-gauss-sigma`, `--return-floor`, and `--models-json`.

## CLI reference

Training options live in `config/arguments.py` (e.g. `--use-soft-target`, `--td-loss`, `--rl-model`, `--training-epochs`, `--model-load-path`).

## Citation

CMIX (baseline idea and constrained setting):

```bibtex
@inproceedings{liu2021cmix,
  title={CMIX: Deep Multi-agent Reinforcement Learning with Peak and Average Constraints},
  author={Liu, Chenyi and Nan Geng and Vaneet Aggarwal and Tian Lan and Yuan Yang and Mingwei Xu},
  booktitle={Proc. ECML-PKDD},
  year={2021}
}
```

Questions about the original CMIX release: see the paper or contact the authors listed there.
