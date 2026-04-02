"""Runnable setup verification: python -m rag_evo.utils.verify_setup"""

import sys
from datetime import datetime

from rag_evo.utils.logger import get_logger
from rag_evo.utils.paths import get_project_paths
from rag_evo.utils.seed import set_global_seed


def main() -> int:
    try:
        paths = get_project_paths()

        missing = [
            str(p)
            for name, p in paths.__dict__.items()
            if not p.exists()
        ]
        if missing:
            print(f"ERROR: Missing directories: {missing}", file=sys.stderr)
            return 1

        set_global_seed(42)

        run_id = "verify_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        log = get_logger("verify", paths.logs, run_id)
        log.info("RAG-Evo setup verified successfully.")

        log_file = paths.logs / f"{run_id}_verify.log"
        if not log_file.exists():
            print(f"ERROR: Log file not created at {log_file}", file=sys.stderr)
            return 1

        print(f"Setup verified. Log written to: {log_file}")
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
