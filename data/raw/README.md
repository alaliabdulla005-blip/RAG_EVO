# Dataset inputs

The dissertation evaluates HotpotQA and CUAD QA. The current loaders retrieve the datasets through Hugging Face and cache the downloaded inputs locally.

| Track | Source and split | Loader |
| --- | --- | --- |
| General QA | HotpotQA distractor validation split | [HotpotQA loader](../../src/rag_evo/data/hotpotqa_loader.py) |
| Legal QA | `chenghao/cuad_qa`, test split | [CUAD loader](../../src/rag_evo/data/legal_loader.py) |

Run the following from the repository root after installing the package:

```bash
python scripts/rebuild_debug_subsets.py
```

The [rebuild script](../../scripts/rebuild_debug_subsets.py) prepares 500 HotpotQA records and 400 CUAD records under `data/processed/`. It replaces the derived debug subsets and removes its temporary backups after success. Phase 16 uses records 0–49 for development in both tracks, records 100–399 for general-QA evaluation and records 50–349 for legal evaluation.

The CUAD loader skips unanswerable records and truncates contract text before building the corpus. These choices define the evaluation subset. Keep original dataset downloads separate from derived files.
