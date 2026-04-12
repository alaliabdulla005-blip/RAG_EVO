# RAG-Evo: Evolutionary Optimisation of RAG for General QA and Legal Queries

Offline research framework using a scalar-weighted genetic algorithm to optimise
Retrieval-Augmented Generation pipelines. Evaluation targets: HotpotQA (general QA)
and CUAD / ContractNLI (legal QA). Generator: Llama-3.1-8B-Instruct via HuggingFace
Inference Endpoints.

---

## Setup

```bash
git clone https://github.com/psyaa33/RAG_EVO.git
cd RAG_EVO
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
cp .env.example .env
# Edit .env and add your HF_TOKEN
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
- `.env` with `INFERENCE_PROVIDER=sambanova` and `SAMBANOVA_API_KEY`
- Debug subsets present in `data/processed/` (150 records each)

### Development set baselines (records 0-49)

```bash
python scripts/run_baseline_comparison.py
python scripts/run_legal_baselines.py
python scripts/run_iterative_rag.py
python scripts/run_iterative_rag_legal.py
python scripts/run_cross_track_transfer.py
```

### Held-out evaluation (records 50-149)

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
