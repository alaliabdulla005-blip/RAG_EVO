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
