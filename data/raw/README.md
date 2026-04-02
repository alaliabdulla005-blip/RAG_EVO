# data/raw — Immutable Raw Data

This directory is **immutable**. Files here must never be modified, overwritten,
or deleted once downloaded. All transformations produce outputs in `data/processed/`.

All files in this directory are gitignored. You must download them locally before
running any data ingestion phases.

---

## Datasets

### 1. HotpotQA (distractor setting)

- **Source:** https://hotpotqa.github.io
- **Download instructions:**
  ```bash
  mkdir -p data/raw/hotpotqa
  curl -L http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json \
       -o data/raw/hotpotqa/hotpot_train_v1.1.json
  curl -L http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json \
       -o data/raw/hotpotqa/hotpot_dev_distractor_v1.json
  ```
- **Setting:** Use the **distractor** dev set for evaluation (not the fullwiki setting).

---

### 2. CUAD — Contract Understanding Atticus Dataset

- **Source:** https://huggingface.co/datasets/cuad
- **Download instructions:**
  ```bash
  mkdir -p data/raw/cuad
  python - <<'EOF'
  from datasets import load_dataset
  ds = load_dataset("cuad")
  ds.save_to_disk("data/raw/cuad")
  EOF
  ```
- Requires `pip install datasets` (install into your venv first).

---

### 3. ContractNLI

- **Source:** https://stanfordnlp.github.io/contract-nli
- **Download instructions:**
  ```bash
  mkdir -p data/raw/contractnli
  # Download from the Stanford NLP page linked above — direct download link is
  # provided on the dataset page as a zip archive.
  # Extract into data/raw/contractnli/
  ```

---

## Notes

- All files in this directory are excluded from version control via `.gitignore`.
- The `.gitkeep` file is the only tracked file here.
- Download before beginning Phase 2 (data ingestion).
