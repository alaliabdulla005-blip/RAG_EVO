# RAG-Evo: Evolutionary Optimisation of RAG for General QA and Legal Queries

Offline research framework using a scalar-weighted genetic algorithm to optimise
Retrieval-Augmented Generation pipelines. Evaluation targets: HotpotQA (general QA)
and CUAD (legal QA). Generator: Llama-3.1-8B-Instruct served via
DeepInfra (primary, Phase 16) or SambaNova Cloud (legacy, Phases 10–15). Provider
selected via the INFERENCE_PROVIDER environment variable.

---

## Setup

```bash
# Requires Python 3.10 or newer (tested on 3.10, 3.11)
git clone https://github.com/psyaa33/RAG_EVO.git
cd RAG_EVO
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
cp .env.example .env
# Edit .env and fill in ONE of the following:
#   INFERENCE_PROVIDER=deepinfra  + DEEPINFRA_API_KEY=<your key>
# Note: SambaNova is no longer supported — their
# Llama-3.1-8B-Instruct endpoint was deprecated
# April 2026.
python -m rag_evo.utils.verify_setup
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Code quality

```bash
ruff check . --fix
```

---

## Reproducing dissertation results

All evaluation scripts require:
- Package installed in dev mode (`pip install -e .`)
- `.env` with `INFERENCE_PROVIDER=deepinfra` and `DEEPINFRA_API_KEY` set — SambaNova is no longer supported as their Llama-3.1-8B-Instruct endpoint was deprecated in April 2026
- Debug subsets present in `data/processed/` (150 records each)

### Development set baselines (records 0-49)

```bash
python scripts/run_baseline_comparison.py
python scripts/run_legal_baselines.py
python scripts/run_iterative_rag.py
python scripts/run_iterative_rag_legal.py
python scripts/run_cross_track_transfer.py
```

### Held-out evaluation (records 100–399 for General-QA and records 50–349 for Legal (n=300 per track))

```bash
python scripts/run_held_out_complete.py
```

### GA search (records 0-49)

```bash
python scripts/run_phase12_general_qa.py
python scripts/run_phase13_legal_qa.py
```

### Ablations

```bash
python scripts/run_seed_stability.py
python scripts/run_mutation_ablation.py
python scripts/run_population_ablation.py
python scripts/run_temperature_ablation_legal.py
```

### Figures and tables

```bash
python scripts/generate_dissertation_figures.py
python scripts/generate_cross_track_table.py
```

### Output locations

| Script | Output |
|--------|--------|
| `run_baseline_comparison.py` | `results/ablations/baseline_comparison.json` |
| `run_legal_baselines.py` | `results/phase14/legal_baselines.json` |
| `run_iterative_rag.py` | `results/ablations/iterative_rag.json` |
| `run_iterative_rag_legal.py` | `results/phase13/iterative_rag_legal.json` |
| `run_cross_track_transfer.py` | `results/phase14/cross_track_transfer.json` |
| `run_held_out_complete.py` | `results/phase15/held_out_complete.json` |

---

## Phase 16 — EC Extension (DeepInfra)

SambaNova deprecated Meta-Llama-3.1-8B-Instruct on 2026-04-14. Phase 16
migrates to DeepInfra (FP8, $0.02/M input, no daily token ceiling) with an
8-gene `FullV2Config` chromosome and 3 independent seeds on General-QA and 1 seed on Legal (seeds 123 and 7 for Legal were deferred due to API budget constraints following the provider migration).

### Provider setup

```bash
# Set INFERENCE_PROVIDER=deepinfra in .env
# Add DEEPINFRA_API_KEY (get from deepinfra.com/dash/api_keys)
# RAGAS faithfulness judge uses Llama-3.3-70B-Instruct-Turbo
# via DeepInfra — no separate key needed,
# DEEPINFRA_API_KEY is reused.
```

### Reproduction steps (in order)

```bash
# 1. Provider calibration (verify DeepInfra ≈ SambaNova ±3%)
INFERENCE_PROVIDER=deepinfra python scripts/run_provider_calibration.py

# 2. GA search — General-QA (3 seeds, run sequentially or in parallel)
INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_general_qa_seed42.py
INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_general_qa_seed123.py
INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_general_qa_seed7.py

# 3. GA search — Legal (3 seeds)
INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_legal_seed42.py
# Note: Legal seeds 123 and 7 were not completed
# due to API budget constraints following the
# provider migration. Only seed 42 is reproduced.

# 4. Held-out evaluation (all systems, BERTScore)
INFERENCE_PROVIDER=deepinfra python scripts/run_held_out_phase16.py

# 5. RAGAS Faithfulness (Llama-3.3-70B-Instruct-Turbo via DeepInfra)
python scripts/run_ragas_faithfulness.py

# 6. Statistical analysis
python scripts/run_statistical_analysis.py

# 7. Generate figures
python scripts/generate_phase16_figures.py
```

### Expected costs

| Component | Est. tokens | Cost |
|-----------|------------|------|
| 3 QA GA runs (pop=50 x gen=100) | ~1.5B input | ~$30 |
| 1 Legal GA run (pop=50 x gen=100) | ~200M | ~$4 |
| Held-out evaluation (n=300 per track) | ~500M | ~$10 |
| RAGAS Faithfulness (DeepInfra 70B) | ~50M | ~$1 |
| **Total** | | **~$45** |

### Expected wall clock

~8 hours total (3 QA GA runs at ~2h each, 1 Legal GA run at ~2h).
Checkpoint/resume: safe to interrupt and restart any GA script.

### Phase 16 output locations

| Script | Output |
|--------|--------|
| `run_provider_calibration.py` | `results/phase16/calibration.json` |
| `run_phase16_general_qa_seed*.py` | `results/phase16/general_qa/seed_*/` |
| `run_phase16_legal_seed*.py` | `results/phase16/legal/seed_*/` |
| `run_held_out_phase16.py` | `results/phase16/held_out_phase16.json` |
| `run_ragas_faithfulness.py` | `results/phase16/ragas_faithfulness.json` |
| `run_statistical_analysis.py` | `results/phase16/statistical_analysis.json` |
| `generate_phase16_figures.py` | `results/phase16/figures/*.png` |

---

## Project structure

```
src/rag_evo/        # Installable package
configs/            # YAML experiment configurations
data/raw/           # Immutable raw datasets (gitignored, download locally)
data/processed/     # Transformed data
data/indexes/       # Vector index artifacts
results/            # Experiment output JSON artifacts
logs/               # Timestamped run logs
tests/              # Pytest test suite
docs/               # Implementation contract and design notes
notebooks/          # Exploratory notebooks
scripts/            # One-off utility scripts
```

---

## Key constraints

- Generator model is fixed: Llama-3.1-8B-Instruct. No alternatives.
- `citations_required` is always `True` — it is a constant, not a parameter.
- `top_p` is fixed at `1.0` — not tunable.
- Raw data in `data/raw/` is immutable. Never write to it.
- All random operations must use `set_global_seed()` for reproducibility.
