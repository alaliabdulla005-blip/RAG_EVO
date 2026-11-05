# RAG-Evo

RAG-Evo is a Python research framework for evolutionary tuning of retrieval-augmented generation (RAG). It searches retrieval, prompting and decoding settings for general question answering on HotpotQA and contract question answering on CUAD. The generator is Llama-3.1-8B-Instruct.

## Experiment design

The genetic algorithm ranks configurations by a weighted combination of token-level answer F1, citation F1 and retrieval recall, with penalties for input tokens and latency. The legal track gives citation F1 the highest weight. [The fitness implementation](src/rag_evo/ga/fitness.py) clips the penalties at 32,500 total input tokens per evaluation and 5,000 ms mean latency.

Phases 12 and 13 used 20 candidates over 15 generations. Phase 16 used 50 candidates over 100 generations. The final chromosome had eight fields, of which six varied: chunk size, overlap, retrieval depth, prompt strategy, temperature and maximum output tokens. Retrieval stayed fixed to BM25 and reranking stayed off.

| Phase 16 track | Dataset | Development questions | Held-out questions | Completed seeds |
| --- | --- | ---: | ---: | --- |
| General QA | HotpotQA, distractor setting | 50 | 300 | 42, 123, 7 |
| Legal QA | CUAD QA | 50 | 300 | 42 |

The development set uses records 0–49. The final held-out sets use records 100–399 for general QA and 50–349 for legal QA. Earlier held-out comparisons used 100 questions per track.

## Saved results

Composite fitness from the [Phase 16 held-out results](results/phase16/held_out_phase16.json), rounded to three decimal places:

| System | HotpotQA | CUAD QA |
| --- | ---: | ---: |
| Direct LLM | −0.051 | −0.032 |
| Naive RAG | 0.155 | −0.027 |
| Conservative RAG | 0.134 | −0.027 |
| Iterative RAG | 0.195 | −0.045 |
| Earlier native GA configuration (Phase 12 / 13) | 0.233 | 0.053 |
| Phase 16 native GA configurations | 0.274–0.278 | 0.053 |
| Phase 16 configuration transferred from the other track | 0.270 | 0.012–0.020 |

On HotpotQA, the saved Phase 16 native configurations score about 77–79% above Naive RAG on composite fitness. On CUAD, the earlier and later native GA configurations both round to 0.053. The earlier one is slightly higher at full precision. The sign of the legal baseline scores depends on the fitness formula. Recomputing with mean input tokens divided by 10,000, as described in the dissertation, would put three of the four baselines above zero.

The [extended metrics](results/phase16/figures/phase16_extended_metrics.png) show a trade-off. On general QA, the seed-42 configuration scores 0.744 for [custom LLM-judged faithfulness](results/phase16/ragas_faithfulness.json), compared with 0.986 for Naive RAG. The judge is Llama-3.3-70B-Instruct-Turbo, with 50 samples per selected system. The [judge script](scripts/run_ragas_faithfulness.py) uses a custom scoring prompt. BERTScore was computed separately on the held-out runs.

## Provider change and interpretation

Phase 16 moved from SambaNova to DeepInfra after the original generator endpoint was retired. On the [50-question calibration](results/phase16/calibration.json), retrieval recall stayed at 0.730 and citation F1 fell from 0.437 to 0.426, a relative change of −2.59%. Answer F1 fell from 0.0587 to 0.0368 and mean latency rose from about 1.04 s to 2.08 s. The overall calibration verdict was a failure.

The dissertation and presentation report $50.31 in total inference spend. The presentation reports about 95 hours for the full experimental programme. Separately, the logged runtimes of the four Phase 16 search runs add up to about 50 hours. Fresh costs and runtimes depend on provider rates, hardware and cache use.

- The Direct LLM run used a context-only prompt without retrieved context. Its scores describe a context-constrained comparison.
- The held-out entries for QA seeds 123 and 7 have different configurations and identical per-record outputs on both evaluation tracks. The runner can reuse cache entries across settings rounded to the same key. Treat the matching entries on each track as a single set of held-out predictions when interpreting stability.
- The CUAD loader filters for answerable questions and truncates contract text. The citation metric matches document identifiers. Clause-level support is a separate question.
- Legal Phase 16 results come from one completed search seed. The final held-out data and code support descriptive comparisons. The statistical script's held-out reader expects a different layout from the saved results.

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . bm25s pytest ruff
python -m pytest tests/ -v
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`. The current BM25 implementation imports `bm25s`, which needs to be installed separately from the package requirements.

## Inspect or rerun the experiments

The saved results and [figures](results/phase16/figures/) can be inspected without a provider key. To regenerate the Phase 16 figures from the saved JSON files, run:

```bash
python scripts/generate_phase16_figures.py
```

A fresh Phase 16 run requires provider credentials and access to the model. Copy [`.env.example`](.env.example) to `.env` and set `INFERENCE_PROVIDER=deepinfra` and `DEEPINFRA_API_KEY` locally. Prepare the [data subsets](data/raw/README.md) in a fresh checkout, and install `bert-score` for the extended held-out evaluation.

```bash
python scripts/rebuild_debug_subsets.py
python scripts/run_phase16_general_qa_seed42.py
python scripts/run_phase16_general_qa_seed123.py
python scripts/run_phase16_general_qa_seed7.py
python scripts/run_phase16_legal_seed42.py
python scripts/run_held_out_phase16.py
python scripts/run_ragas_faithfulness.py
```

The rebuild script replaces the derived debug subsets and removes its temporary backups after success. The search scripts use checkpoints. The held-out and judge scripts use saved outputs to resume, so use a separate working copy if you need to preserve the original artifacts or run a fresh comparison. The judge script requires per-system pipeline outputs with retrieved context. Provider behaviour and cache contents can affect reruns.
