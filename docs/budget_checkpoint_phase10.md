# Budget Checkpoint — Phase 10 GA Pilot (Updated with Actuals)

Date: 08 April 2026

---

## Estimated vs Actual

| Item | Estimated | Actual |
|------|-----------|--------|
| Total API calls | 7,500 | ~5,385 (cache reduced unique calls) |
| Input tokens | ~8.25M | **8,078,295** (SambaNova dashboard) |
| Output tokens | ~375K | **189,155** (SambaNova dashboard) |
| Input token cost | ~$0.83 | **~$0.81** |
| Output token cost | ~$0.08 | **~$0.04** |
| Total cost | ~$0.90 | **~$0.85** |
| Wall clock time | ~5 hours | **~5 hours active** (17,982s logged elapsed) |
| Provider | SambaNova Developer | SambaNova Developer ✓ |

---

## What actually happened

The pilot ran all 15 generations and completed at approximately 06:51 on 8 April 2026. The process continued running for approximately 3 additional hours after the GA finished — likely due to a post-GA artifact-saving or evaluation step in run_ga_pilot.py that either hung or made additional API calls. The process was killed at approximately 10:00, at which point the completed summary was printed from memory. The GA results are genuine — generation 13's 0.24-second completion time is physically impossible without full cache hits, confirming both the result and index caches were working.

**Rate limit issue encountered:** SambaNova's free tier is capped at 20 requests per day, which blocked the pilot entirely. Resolution: upgraded to SambaNova Developer tier. Actual daily usage peaked at 8,078,295 input tokens on 8 April — well within the Developer tier's 20M token/day limit.

---

## Actual per-generation costs

| Phase | Generations | Avg elapsed | Cache behaviour |
|-------|-------------|-------------|-----------------|
| Exploration (Gen 0–4) | 5 | 2,044s | Mostly misses — new configs |
| Convergence (Gen 5–12) | 8 | 737s | Increasing hits |
| Full convergence (Gen 13) | 1 | 0.24s | All 50 records cached |
| Late (Gen 14) | 1 | 970s | Mixed — new mutations |

---

## Decision for Phase 12 full experiments

Proceed, with the following adjustments based on pilot learnings:

1. **Population size: increase from 10 to 20.** The pilot showed premature convergence from generation 5 onwards — a ten-generation plateau with no improvement. Larger population maintains genetic diversity longer.
2. **Generations: keep at 15.** If stagnation (no best improvement for 3+ generations) is detected, activate restart mechanism.
3. **Records: keep 50 for full Phase 12 run.** Long generation 0 time was due to all cache misses; subsequent generations will be faster as the cache fills.
4. **Fix post-GA process hang.** Identify and fix the code in run_ga_pilot.py that kept the process alive for 3 hours after the GA completed.
5. **Estimated Phase 12 cost:** 20 individuals × 50 records × 15 generations = up to 15,000 unique calls. With cache reducing actual calls to ~60%: ~9,000 calls × 1,500 avg tokens = ~13.5M input tokens × $0.10/M = **~$1.35 total**.
