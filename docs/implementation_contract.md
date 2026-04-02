# Implementation Contract — Quick Reference

See `docs/Implementation_Contract.pdf` for the full contract.

---

## Six Fixed Decisions

1. **Generator model is fixed.**
   Llama-3.1-8B-Instruct served via a dedicated HuggingFace Inference Endpoint.
   No model swaps, alternatives, or local inference.

2. **Evaluation tracks are sequential.**
   Phase 1: General QA on HotpotQA (distractor setting).
   Phase 2: Legal QA on CUAD and ContractNLI.
   Results from Phase 1 inform Phase 2 configuration but do not carry over directly.

3. **Optimisation algorithm is a scalar-weighted genetic algorithm.**
   The decision vector is mixed-type (continuous + discrete parameters).
   No other optimisation strategy will be used or compared.

4. **`citations_required` is always True.**
   It is a constant, not a configurable parameter. Do not expose it as a variable.

5. **`top_p` is fixed at 1.0.**
   It is not a tunable parameter and must not be added to the decision vector
   or any configuration schema.

6. **Every experiment run produces a timestamped JSON artifact.**
   In-memory state must never be the sole record of an experiment result.
   Artifacts are written to `results/` before any downstream processing.
