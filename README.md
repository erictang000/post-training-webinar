# Post-training with Tinker: live demo

One idea for the whole talk: **every method is the same loop** (`forward_backward` → `optim_step` → `sample`).
Methods differ only in *where the tokens come from* and *what signal each token gets*.

| file | method | token source | per-token signal | Tinker loss |
|---|---|---|---|---|
| `00_setup.ipynb` | baseline | — | — | — |
| `01_pretrain.ipynb` | pre-training objective | raw text (Shakespeare) | 1 everywhere | `cross_entropy` |
| `02_sft.ipynb` | SFT | GSM8K demos | 0 prompt / 1 response | `cross_entropy` |
| `03_dpo.ipynb` | DPO | chosen/rejected pairs | log-ratio vs. reference | `forward_backward_custom` |
| `04_rl.ipynb` | RL (GRPO, from `rl_loop`) | **sampled from policy** | reward − group mean | `importance_sampling` |
| `05_distillation.ipynb` | OPD / OPSD / MOPD | **sampled from student** | teacher lp − student lp | `importance_sampling` |

Student: `Qwen/Qwen3.5-4B` (smallest model on Tinker), thinking disabled. Task: GSM8K answered correctly **within 256 tokens**. The model already knows the math but writes long answers, so post-training teaches it to be correct and concise. Scoreboard = first 100 GSM8K test problems. Teachers: `Qwen/Qwen3.6-35B-A3B` (math), `Qwen/Qwen3.5-397B-A17B` (chat). They have to share the student's tokenizer.

## Setup
Dependencies live in `pyproject.toml`, pinned by `uv.lock`. One command builds `.venv`:
```bash
uv sync
```
Then register it as a Jupyter kernel (once per machine):
```bash
uv run python -m ipykernel install --user --name webinar --display-name "Python (webinar .venv)"
```
The API key is read from `~/.tinker_api_key` (or `$TINKER_API_KEY`).

Open the notebooks with the **Python (webinar .venv)** kernel and run them top to bottom. In VS Code / Cursor the kernel picker also lists `.venv` directly under this folder; either entry is the same interpreter.

To add a dependency later, use `uv add <pkg>` — it updates `pyproject.toml`, `uv.lock` and `.venv` together (don't `pip install` into `.venv` by hand, or the lock drifts).

Cells marked `── LIVE ──` are the ones to type on stage. Everything else is pre-written, and `common.py` holds the plumbing.

## Chaining and backups
- `02_sft.ipynb` saves an `sft` checkpoint. `04_rl.ipynb` and `05_distillation.ipynb` start from it automatically, or from the base model if it doesn't exist.
- Checkpoint paths and GSM8K scores are written to `checkpoints.json`. Run everything once before the webinar so you have a full scoreboard and fallback checkpoints. `scoreboard()` prints it.
