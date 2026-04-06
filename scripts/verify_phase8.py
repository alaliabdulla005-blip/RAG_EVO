"""Phase 8 verification — config schema and search space.
Run from project root: python scripts/verify_phase8.py
"""

import random
from pathlib import Path

from rag_evo.config.schema import FullV1Config, MicroPilotConfig
from rag_evo.config.space import SearchSpace
from rag_evo.config.validators import (
    is_valid_full_v1,
    is_valid_micro_pilot,
    repair_micro_pilot,
)

print("=== Phase 8 — Config Schema Verification ===\n")

# 1. Default configs
micro = MicroPilotConfig()
full = FullV1Config()
print(f"MicroPilotConfig default: {micro.to_dict()}")
print(f"FullV1Config default:     {full.to_dict()}\n")

# 2. Fixed constants
assert micro.citations_required is True, "citations_required must be True"
assert micro.top_p == 1.0, "top_p must be 1.0"
assert full.citations_required is True
assert full.top_p == 1.0
print("Fixed constants confirmed: citations_required=True, top_p=1.0\n")

# 3. JSON roundtrip
out_dir = Path("results/configs/phase8_verify")
out_dir.mkdir(parents=True, exist_ok=True)
micro.to_json(out_dir / "micro_default.json")
full.to_json(out_dir / "full_default.json")
micro2 = MicroPilotConfig.from_json(out_dir / "micro_default.json")
full2 = FullV1Config.from_json(out_dir / "full_default.json")
assert micro.to_dict() == micro2.to_dict(), "Micro roundtrip failed"
assert full.to_dict() == full2.to_dict(), "Full roundtrip failed"
print("JSON roundtrip: PASSED\n")

# 4. Search space sampling
rng = random.Random(42)
micro_space = SearchSpace("micro_pilot")
full_space = SearchSpace("full_v1")

micro_samples = [micro_space.sample_random(rng) for _ in range(100)]
full_samples = [full_space.sample_random(rng) for _ in range(100)]

micro_valid = all(is_valid_micro_pilot(s) for s in micro_samples)
full_valid = all(is_valid_full_v1(s) for s in full_samples)
print(f"100 micro_pilot samples all valid: {micro_valid}")
print(f"100 full_v1 samples all valid:     {full_valid}\n")

# 5. Gene diversity check
chunk_sizes = set(s.chunk_size for s in micro_samples)
strategies = set(s.prompt_strategy for s in micro_samples)
retriever_types = set(s.retriever_type for s in full_samples)
print(f"Chunk sizes seen in 100 samples:   {sorted(chunk_sizes)}")
print(f"Strategies seen in 100 samples:    {sorted(strategies)}")
print(
    f"Retriever types seen in 100 samples: {sorted(retriever_types)}\n"
)

# 6. Repair functions
invalid_micro = MicroPilotConfig(
    chunk_size=300, overlap=0.15, top_k=25, temperature=1.5
)
repaired = repair_micro_pilot(invalid_micro)
print("Invalid micro:  chunk_size=300, overlap=0.15, top_k=25, temp=1.5")
print(
    f"Repaired micro: chunk_size={repaired.chunk_size}, "
    f"overlap={repaired.overlap}, "
    f"top_k={repaired.top_k}, "
    f"temp={repaired.temperature}"
)
assert is_valid_micro_pilot(repaired), "Repaired config must be valid"
print("Repair function: PASSED\n")

# 7. copy_with
cfg = MicroPilotConfig(chunk_size=256, top_k=5)
new_cfg = cfg.copy_with(chunk_size=512, top_k=10)
assert new_cfg.chunk_size == 512
assert new_cfg.top_k == 10
assert cfg.chunk_size == 256  # original unchanged
print("copy_with: PASSED\n")

# 8. Gene names
print(f"Micro-pilot gene names: {micro_space.get_gene_names()}")
print(f"Full v1 gene names:     {full_space.get_gene_names()}\n")

print("=== PASTE THIS SUMMARY FOR ANALYSIS ===")
print(f"Micro valid: {micro_valid}")
print(f"Full valid:  {full_valid}")
print(f"Chunk sizes seen: {sorted(chunk_sizes)}")
print(f"Strategies seen:  {sorted(strategies)}")
print(f"Retriever types:  {sorted(retriever_types)}")
print(
    f"Repair test:  chunk_size={repaired.chunk_size} "
    f"overlap={repaired.overlap} "
    f"top_k={repaired.top_k} "
    f"temp={repaired.temperature}"
)
print("Phase 8 verification complete.")
