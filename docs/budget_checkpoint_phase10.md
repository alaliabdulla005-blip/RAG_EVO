# Budget Checkpoint — Before Phase 10

Date: 2026-04-07

## Measured Actuals (n=10, BM25, zero-shot, Groq free tier)
- Mean input tokens per query: 626.3
- Mean output tokens per query: 22.3
- Mean tokens per query (total): 648.6
- True query latency (no rate limit): ~150 ms
- Rate-limited latency (Groq free): 2,475 ms mean

## Groq Free Tier Rate Limits (confirmed from dashboard)
- Tokens per minute: 6,000 (BINDING constraint)
- Requests per day: 14,400
- Tokens per day: 500,000
- Effective throughput: 9.2 queries/minute (token-limited)
- Days needed for full pilot (worst case): 9.7 days
- Verdict: INFEASIBLE — exceeds 8-day deadline

## Provider Evaluation Summary
All providers evaluated before selecting SambaNova:

| Provider | Precision | Cost | Verdict |
|---|---|---|---|
| Groq free | TruePoint quantised | $0 | Too slow for GA |
| Groq Developer | TruePoint quantised | ~$2.55 | Precision issue vs literature |
| Together AI Turbo | FP8 | ~$3.30 | Not full precision |
| Together AI AWQ Int4 | 4-bit | Cheapest | Significantly degraded |
| Together AI BF16 | Full BF16 | N/A | Does not exist at serverless tier |
| HuggingFace API | Routes to providers above | $9/month+ | Not independent |
| Local vLLM | Full BF16 | $0 | Requires 16 GB — only 6 GB available |
| **SambaNova Cloud** | **Full BF16/FP32** | **$0 free tier** | **SELECTED** |

## Selected Provider: SambaNova Cloud
- Model: Meta-Llama-3.1-8B-Instruct
- Precision: Full BF16/FP32 mixed (confirmed by official docs + independent benchmarks)
- Matches literature: Yes — Orbach et al. (2025) and Park et al. (2025) use full-precision Llama-3.1-8B-Instruct
- Free tier: 30 RPM permanent (no expiry)
- Signup credit: $5 (~30M tokens on Llama 8B, ~3 month expiry)
- Base URL: https://api.sambanova.ai/v1
- API compatibility: OpenAI-compatible

## Full Dissertation Cost Projection (SambaNova)
| Phase | Calls (realistic) | Tokens (est.) | Cost |
|---|---|---|---|
| Phase 10 — GA pilot | 1,200 | 778,320 | $0.08 |
| Phase 12 — Full General-QA | 6,000 | 3,891,600 | $0.39 |
| Phase 14 — Legal track | 3,000 | 1,945,800 | $0.19 |
| **Total** | **10,200** | **6,615,720** | **$0.67** |

Covered by $5 signup credit with $4.33 remaining.

## GA Pilot Parameters (full quality — no reduction)
- Population: 10
- Generations: 15
- Records per evaluation: 50
- Total calls (realistic, 20% cache): 6,000
- Time per generation at 30 RPM: ~17 minutes active querying
- Timeline: Feasible within deadline with per-generation checkpointing

## Two-Tier Provider Strategy
Development (Phases 1–9): Groq free tier (llama-3.1-8b-instant, TruePoint quantised)
  — Used for pipeline construction, interface validation, smoke tests
  — Zero cost, permanent free tier
  — Development metrics are NOT reported results

Experiments (Phase 10+): SambaNova Cloud (Meta-Llama-3.1-8B-Instruct, full BF16/FP32)
  — All reported results produced here
  — Matches literature comparator precision

## Phase 10 Requirements
1. Extend GeneratorWrapper to support SambaNova via OpenAI-compatible endpoint
2. INFERENCE_PROVIDER env variable switches between sambanova/groq
3. Re-run Phase 6 baseline (10 queries) on SambaNova for official baseline metrics
4. Per-generation checkpointing mandatory
5. Index caching keyed by (chunk_size, overlap, retriever_type)

## Verdict
FULLY FEASIBLE at full quality parameters.
Total cost: $0.67 — covered by $5 signup credit.
Timeline: ~17 min active querying per generation, spread across sessions.