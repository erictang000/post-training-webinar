# Post-training with Tinker: live demo

One idea for the whole talk: **every method is the same loop** (`forward_backward` → `optim_step` → `sample`).
Methods differ only in *where the tokens come from* and *what signal each token gets*.

| file | method | token source | per-token signal | Tinker loss |
|---|---|---|---|---|
| `00_setup.ipynb` | baseline | — | — | — |
| `01_pretrain.ipynb` | pre-training objective | raw text (Shakespeare) | 1 everywhere | `cross_entropy` |
| `02_sft.ipynb` | SFT | GSM8K demos | 0 prompt / 1 response | `cross_entropy` |
| `03_rl.ipynb` | RL (GRPO, from `rl_loop`) | **sampled from policy** | reward − group mean | `importance_sampling` |
| `04_distillation.ipynb` | OPD / OPSD | **sampled from student** | teacher lp − student lp | `importance_sampling` |

Student: `Qwen/Qwen3.5-4B` (smallest model on Tinker), thinking disabled. Task: GSM8K answered correctly **within 256 tokens**. The model already knows the math but writes long answers, so post-training teaches it to be correct and concise. Scoreboard = first 100 GSM8K test problems. OPD teacher: `Qwen/Qwen3.6-35B-A3B`. It has to share the student's tokenizer. The OPSD teacher is a frozen copy of the student.

## Setup
Dependencies live in `pyproject.toml`, pinned by `uv.lock`. One command builds `.venv`:
```bash
uv sync
```
Then register it as a Jupyter kernel (once per machine):
```bash
uv run python -m ipykernel install --user --name webinar --display-name "Python (webinar .venv)"
```
The API key is read from, in order: `$TINKER_API_KEY`, a `.env` file in this folder, or `~/.tinker_api_key`. The easiest option is `.env`, which is gitignored:
```bash
echo 'TINKER_API_KEY=tml-...' > .env
```
Jupyter kernels inherit their environment from whatever launched Jupyter (VS Code, JupyterLab), so an `export` in a terminal won't reach an already-running kernel. Use one of the files instead, then restart the kernel.

Open the notebooks with the **Python (webinar .venv)** kernel and run them top to bottom. In VS Code / Cursor the kernel picker also lists `.venv` directly under this folder; either entry is the same interpreter.

To add a dependency later, use `uv add <pkg>` — it updates `pyproject.toml`, `uv.lock` and `.venv` together (don't `pip install` into `.venv` by hand, or the lock drifts).

Cells marked `── LIVE ──` are the ones to type on stage. Everything else is pre-written, and `common.py` holds the plumbing.

## Chaining and backups
- `02_sft.ipynb` saves an `sft` checkpoint. `03_rl.ipynb` and `04_distillation.ipynb` start from it automatically, or from the base model if it doesn't exist.
- Checkpoint paths and GSM8K scores are written to `checkpoints.json`. Run everything once before the webinar so you have a full scoreboard and fallback checkpoints. `scoreboard()` prints it.

## Part 2: the RL notebook on your own GPUs (SkyRL)
`03_rl.ipynb` also runs against a [SkyRL](https://github.com/NovaSky-AI/SkyRL) Tinker server on this node (4×H100, FSDP, colocated training + vLLM). SkyRL is cloned and installed at `~/default/skyrl-webinar` (`uv sync --extra tinker --extra fsdp`).
1. Run the server cell at the bottom of `03_rl.ipynb`. It (re)starts the server in about 10 s.
2. At the top, set `BASE_URL = "http://localhost:8000"`. That also cuts the run to 5 steps.
3. Re-run from the top. The first step takes ~3 min longer while SkyRL loads the model onto the GPUs. After that it's ~35 s per step, and about 5–6 min end to end.

On FSDP a SkyRL server holds one LoRA adapter, so restart the server (re-run its cell) before each new run. SkyRL checkpoints and scores are stored under `SkyRL · …` keys, so they never overwrite the Tinker backups. Stop the server with `pkill -f 'skyrl\.tinker\.(api|engine)'`.
