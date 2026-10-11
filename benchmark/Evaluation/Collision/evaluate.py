from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
    from runner import main

    raise SystemExit(main())
