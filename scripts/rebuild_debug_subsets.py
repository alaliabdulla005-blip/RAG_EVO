"""Rebuild the HotpotQA and Legal debug subsets to a larger size.

The GA search used records[0:50] of each debug subset. The held-out
evaluation (scripts/run_held_out_phase16.py) requires records[100:400]
for General-QA and records[50:350] for Legal. The current debug JSONLs
contain only 150 records each, so the held-out slices cannot be formed.

This script regenerates both subsets at sizes that comfortably cover the
held-out ranges:
    - HotpotQA: 500 records (covers records[100:400])
    - Legal:    400 records (covers records[50:350])

The first 50 records (used by the GA as the dev set) are preserved
byte-for-byte — both loaders take the first N records in deterministic
order, so expanding N does not shift earlier indices.

Safety: the new JSONLs are materialised in a temp directory and only
swapped into place after the write completes successfully, so a partial
run cannot leave behind a truncated qa_records.jsonl that would silently
break the GA reproducibility check.

Run from project root:
    python scripts/rebuild_debug_subsets.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from rag_evo.data.hotpotqa_loader import (
    build_debug_subset,
    load_hotpotqa,
    save_records,
)
from rag_evo.data.legal_loader import build_legal_debug_subset
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

GENERAL_QA_N = 500
LEGAL_N = 400


def _first_record_ids(qa_jsonl: Path, n: int) -> list[str]:
    """Return the record_id field of the first n records in a JSONL file."""
    ids: list[str] = []
    with qa_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ids.append(json.loads(line)["record_id"])
    return ids


def _rebuild_hotpotqa(output_dir: Path, cache_dir: Path, n: int) -> None:
    """Rebuild the HotpotQA debug subset in output_dir with n records.

    Writes to a temp directory and swaps into place atomically.
    """
    print(f"Loading HotpotQA validation split (cache={cache_dir})...")
    qa_all, corpus = load_hotpotqa(
        split="validation",
        cache_dir=cache_dir,
    )
    print(f"  got {len(qa_all)} records, {len(corpus)} corpus docs")

    debug_records = build_debug_subset(qa_all, n=n)
    print(f"  taking first {len(debug_records)} records")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "hotpotqa_debug"
        save_records(debug_records, corpus, tmp_path)
        _swap_directory(tmp_path, output_dir)
    print(f"  wrote {output_dir}")


def _rebuild_legal(output_dir: Path, cache_dir: Path, n: int) -> None:
    """Rebuild the Legal (CUAD) debug subset in output_dir with n records.

    build_legal_debug_subset writes directly to output_dir, so we steer
    it at a temp dir first and swap.
    """
    print(f"Loading CUAD (cache={cache_dir})...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "legal_debug"
        build_legal_debug_subset(
            output_dir=tmp_path,
            n=n,
            cache_dir=cache_dir,
        )
        _swap_directory(tmp_path, output_dir)
    print(f"  wrote {output_dir}")


def _swap_directory(src: Path, dst: Path) -> None:
    """Atomically replace dst with src.

    If dst already exists, it is moved aside to dst.backup before the
    new files are moved in. On success, the backup is deleted.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup = dst.with_name(dst.name + ".backup")
    if dst.exists():
        if backup.exists():
            shutil.rmtree(backup)
        dst.rename(backup)
    shutil.move(str(src), str(dst))
    if backup.exists():
        shutil.rmtree(backup)


def _verify(output_dir: Path, min_records: int, label: str) -> None:
    """Assert the rebuilt JSONL has at least min_records lines."""
    jsonl = output_dir / "qa_records.jsonl"
    with jsonl.open("r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    print(f"  {label}: {count} records in {jsonl}")
    if count < min_records:
        raise AssertionError(
            f"{label}: expected >= {min_records} records, got {count}"
        )


def _verify_dev_set_unchanged(
    previous_ids: list[str] | None,
    new_jsonl: Path,
    n_check: int = 50,
) -> None:
    """Confirm the first n_check record IDs match between old and new files.

    This guards the GA dev set (records[0:50]) against drift. If the
    previous file did not exist, skip the check.
    """
    if previous_ids is None:
        print(f"  (no previous file — skipped records[0:{n_check}] diff)")
        return
    new_ids = _first_record_ids(new_jsonl, n_check)
    if previous_ids != new_ids:
        mismatches = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(previous_ids, new_ids))
            if a != b
        ]
        raise AssertionError(
            f"records[0:{n_check}] IDs drifted after rebuild in "
            f"{new_jsonl} — first mismatch: {mismatches[:3]}"
        )
    print(f"  records[0:{n_check}] unchanged (GA dev set preserved)")


def main() -> int:
    paths = get_project_paths()
    qa_out = paths.root / "data" / "processed" / "hotpotqa_debug"
    legal_out = paths.root / "data" / "processed" / "legal_debug"
    qa_cache = paths.root / "data" / "raw" / "hotpotqa_cache"
    legal_cache = paths.root / "data" / "raw" / "cuad_cache"

    qa_prev_ids = (
        _first_record_ids(qa_out / "qa_records.jsonl", 50)
        if (qa_out / "qa_records.jsonl").exists()
        else None
    )
    legal_prev_ids = (
        _first_record_ids(legal_out / "qa_records.jsonl", 50)
        if (legal_out / "qa_records.jsonl").exists()
        else None
    )

    print("=" * 60)
    print("Rebuilding debug subsets")
    print(f"  HotpotQA target: {GENERAL_QA_N} records -> {qa_out}")
    print(f"  Legal target:    {LEGAL_N} records -> {legal_out}")
    print("=" * 60)

    _rebuild_hotpotqa(qa_out, qa_cache, GENERAL_QA_N)
    _verify(qa_out, GENERAL_QA_N, "HotpotQA")
    _verify_dev_set_unchanged(qa_prev_ids, qa_out / "qa_records.jsonl")

    _rebuild_legal(legal_out, legal_cache, LEGAL_N)
    _verify(legal_out, LEGAL_N, "Legal")
    _verify_dev_set_unchanged(legal_prev_ids, legal_out / "qa_records.jsonl")

    print("=" * 60)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
