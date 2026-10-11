from __future__ import annotations

import json
import sys
from pathlib import Path

from stability import evaluate_asset


def main() -> None:
    source, destination = map(Path, sys.argv[1:3])
    result = evaluate_asset(source)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)


if __name__ == "__main__":
    main()
