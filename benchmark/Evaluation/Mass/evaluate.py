from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
    from runner import main

    raise SystemExit(main())
