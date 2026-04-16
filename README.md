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

## Phase 16 — EC Extension (DeepInfra)

SambaNova deprecated Meta-Llama-3.1-8B-Instruct on 2026-04-14. Phase 16
migrates to DeepInfra (FP8, $0.02/M input, no daily token ceiling) with an
8-gene `FullV2Config` chromosome and 3 independent seeds per track.

### Provider setup

```bash
# Set INFERENCE_PROVIDER=deepinfra in .env
# Add DEEPINFRA_API_KEY (get from deepinfra.com/dash/api_keys)
# Add GROQ_API_KEY for RAGAS faithfulness judge
```

### Reproduction steps (in order)

```bash
# 1. Provider calibration (verify DeepInfra ≈ SambaNova ±3%)
INFERENCE_PROVIDER=deepinfra python scripts/run_provider_calibration.py

# 2. Pre-build dense FAISS indexes
python scripts/build_dense_indexes.py

# 3. GA search — General-QA (3 seeds, run sequentially or in parallel)
INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_general_qa_seed42.py
INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_general_qa_seed123.py
INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_general_qa_seed7.py

# 4. GA search — Legal (3 seeds)
INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_legal_seed42.py
INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_legal_seed123.py
INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_legal_seed7.py

# 5. Held-out evaluation (all systems, BERTScore)
INFERENCE_PROVIDER=deepinfra python scripts/run_held_out_phase16.py

# 6. RAGAS Faithfulness (Groq 70B judge)
python scripts/run_ragas_faithfulness.py

# 7. Statistical analysis
python scripts/run_statistical_analysis.py

# 8. Generate figures
python scripts/generate_phase16_figures.py
```

### Expected costs

| Component | Est. tokens | Cost |
|-----------|------------|------|
| 6 GA runs (pop=50 x gen=100) | ~5B input | ~$100 |
| Held-out evaluation | ~500M | ~$10 |
| RAGAS Faithfulness (Groq) | minimal | free tier |
| **Total** | | **~$110** |

### Expected wall clock

~12 hours total (6 GA runs at ~2h each, sequential).
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
