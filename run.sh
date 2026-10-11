#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher=$project_dir/.artisan/bin/artisan
if [[ ! -x $launcher ]]; then
    printf 'Artisan is not installed in this project. Run %s/install.sh first.\n' "$project_dir" >&2
    exit 1
fi
exec "$launcher" "$@"
