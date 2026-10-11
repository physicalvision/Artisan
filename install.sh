#!/usr/bin/env bash
set -Eeuo pipefail

UV_VERSION=0.11.24
BLENDER_VERSION=5.0.0
NODE_VERSION=22.22.1
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

usage() {
    cat <<'EOF'
--yes
--non-interactive
--install-system-deps
--no-system-deps
--offline
--data-dir PATH
--bin-dir PATH
--workspace PATH
--config PATH
--credentials PATH
--uv PATH
--python PATH
--blender PATH
--node PATH
--rebuild-viewer
-h, --help
EOF
}

log() { printf '[Artisan] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
need_value() { [[ $# -ge 2 && -n $2 && $2 != --* ]] || die "$1 requires a value"; }
absolute_path() {
    [[ $1 != *$'\n'* && $1 != *$'\r'* ]] || die 'Paths cannot contain line breaks'
    case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "$PWD" "$1" ;; esac
}

non_interactive=0
system_deps=deny
offline=0
rebuild_viewer=0
data_dir=$source_dir/.artisan
bin_dir=
workspace=
config_path=
credentials=
uv_bin=
python_bin=
blender_bin=
node_bin=
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --yes) non_interactive=1; shift ;;
        --non-interactive) non_interactive=1; shift ;;
        --install-system-deps) system_deps=allow; shift ;;
        --no-system-deps) system_deps=deny; shift ;;
        --offline) offline=1; shift ;;
        --rebuild-viewer) rebuild_viewer=1; shift ;;
        --data-dir) need_value "$@"; data_dir=$2; shift 2 ;;
        --bin-dir) need_value "$@"; bin_dir=$2; shift 2 ;;
        --workspace) need_value "$@"; workspace=$2; shift 2 ;;
        --config) need_value "$@"; config_path=$2; shift 2 ;;
        --credentials) need_value "$@"; credentials=$2; shift 2 ;;
        --uv) need_value "$@"; uv_bin=$2; shift 2 ;;
        --python) need_value "$@"; python_bin=$2; shift 2 ;;
        --blender) need_value "$@"; blender_bin=$2; shift 2 ;;
        --node) need_value "$@"; node_bin=$2; shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die 'Linux x86_64 is required'
[[ -r /etc/os-release ]] || die 'Missing /etc/os-release'
distro=$(. /etc/os-release; printf '%s' "$ID")
distro_version=$(. /etc/os-release; printf '%s' "${VERSION_ID:-0}")
case "$distro" in
    ubuntu) minimum=22.04 ;;
    debian) minimum=12 ;;
    *) die "Unsupported distribution: $distro" ;;
esac
dpkg --compare-versions "$distro_version" ge "$minimum" || die "$distro $minimum or newer is required"

[[ -f $source_dir/uv.lock && -f $source_dir/pyproject.toml ]] || die 'Missing uv.lock or pyproject.toml'
data_dir=$(absolute_path "$data_dir")
bin_dir=$(absolute_path "${bin_dir:-$data_dir/bin}")
config_path=$(absolute_path "${config_path:-$data_dir/config.toml}")
workspace=$(absolute_path "${workspace:-$data_dir/workspace}")
[[ -z $credentials ]] || credentials=$(absolute_path "$credentials")
((offline == 0)) || system_deps=deny

install_system_dependencies() {
    local package glib=libglib2.0-0
    local -a missing=() privileged=()
    if { [[ $distro == ubuntu ]] && dpkg --compare-versions "$distro_version" ge 24.04; } ||
       { [[ $distro == debian ]] && dpkg --compare-versions "$distro_version" ge 13; }; then
        glib=libglib2.0-0t64
    fi
    local -a packages=(ca-certificates curl tar xz-utils util-linux libstdc++6 libgomp1
        libgl1 libegl1 libopengl0 libx11-6 libxext6 libxi6 libxrender1 libxfixes3
        libxxf86vm1 libxkbcommon0 libsm6 libice6 libfontconfig1 "$glib")
    for package in "${packages[@]}"; do
        if ! dpkg-query -W -f='${Architecture}\t${db:Status-Status}\n' "$package" 2>/dev/null |
            awk -F '\t' '($1 == "amd64" || $1 == "all") && $2 == "installed" { found = 1 } END { exit !found }'; then
            missing+=("$package")
        fi
    done
    ((${#missing[@]})) || return 0
    if [[ $system_deps != allow ]]; then
        log "WARNING: Missing system packages: ${missing[*]}"
        log "WARNING: Continuing; install them with 'sudo apt-get install -y ${missing[*]}' or rerun with --install-system-deps"
        return 0
    fi
    log "Missing system packages: ${missing[*]}"
    if ((EUID != 0)); then
        command -v sudo >/dev/null || die 'sudo is unavailable'
        privileged=(sudo)
        ((non_interactive == 0)) || privileged+=(-n)
    fi
    "${privileged[@]}" apt-get update
    "${privileged[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
}
install_system_dependencies

mkdir -p -- "$data_dir/tools" "$bin_dir"
exec 9>"$data_dir/.install.lock"
flock -n 9 || die "Another installer is using $data_dir"
cache_dir=$data_dir/cache/installer
mkdir -p -- "$cache_dir/downloads"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$data_dir/cache"}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-"$data_dir/xdg/config"}
export XDG_DATA_HOME=${XDG_DATA_HOME:-"$data_dir/xdg/data"}
export UV_CACHE_DIR=$data_dir/cache/uv
export npm_config_cache=$data_dir/cache/npm
export npm_config_userconfig=$data_dir/npmrc
export TMPDIR=$data_dir/tmp
export BLENDER_USER_RESOURCES=${BLENDER_USER_RESOURCES:-"$data_dir/cache/blender"}
mkdir -p -- "$TMPDIR" "$data_dir/xdg/config" "$data_dir/xdg/data"
temp_dir=$(mktemp -d "$cache_dir/stage.XXXXXXXX")
cleanup() { rm -rf -- "$temp_dir"; }
trap cleanup EXIT
trap 'log "Installation failed at line $LINENO"' ERR

fetch_archive() {
    local url=$1 filename=$2 file="$cache_dir/downloads/$2"
    if [[ ! -f $file ]]; then
        ((offline == 0)) || die "Offline archive missing: $file"
        log "Downloading $filename"
        curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 \
            --retry 3 --connect-timeout 20 --output "$temp_dir/download" "$url"
        mv -- "$temp_dir/download" "$file"
    fi
    printf '%s\n' "$file"
}

executable_path() {
    local path
    path=$(command -v -- "$1") || die "Executable not found: $1"
    [[ -f $path && -x $path ]] || die "Not an executable file: $path"
    absolute_path "$path"
}

uv_usable() {
    local version
    version=$("$1" --version 2>/dev/null) || return 1
    version=${version#uv }; version=${version%% *}
    [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] && dpkg --compare-versions "$version" ge "$UV_VERSION"
}
if [[ -n $uv_bin ]]; then
    uv_bin=$(executable_path "$uv_bin")
    uv_usable "$uv_bin" || die "--uv must be uv $UV_VERSION or newer"
elif command -v uv >/dev/null && uv_usable "$(command -v uv)"; then
    uv_bin=$(executable_path uv)
else
    uv_bin=$data_dir/tools/uv-$UV_VERSION/uv
    if [[ ! -x $uv_bin ]]; then
        archive=$(fetch_archive "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" uv-$UV_VERSION.tar.gz)
        tar -xf "$archive" -C "$temp_dir" --no-same-owner
        [[ ! -e $data_dir/tools/uv-$UV_VERSION ]] || die 'Incomplete uv installation'
        mv -- "$temp_dir/uv-x86_64-unknown-linux-gnu" "$data_dir/tools/uv-$UV_VERSION"
    fi
    uv_usable "$uv_bin" || die 'Downloaded uv failed its version check'
fi
log "Using $uv_bin"

export UV_PYTHON_INSTALL_DIR=$data_dir/python
export UV_PROJECT_ENVIRONMENT=$data_dir/venv
export UV_NO_PROGRESS=1
uv_network=()
if ((offline)); then uv_network=(--offline); fi
if [[ -n $python_bin ]]; then
    python_bin=$(executable_path "$python_bin")
    "$python_bin" -I -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
else
    log 'Preparing project-managed Python 3.12'
    UV_PYTHON_DOWNLOADS=automatic "$uv_bin" python install 3.12 --no-bin "${uv_network[@]}"
    python_bin=$("$uv_bin" python find --managed-python 3.12 "${uv_network[@]}")
fi
export UV_PYTHON_DOWNLOADS=never

configured_blender=
if [[ -f $config_path ]]; then
    configured_blender=$("$python_bin" -I - "$config_path" <<'PY'
import sys
import tomllib
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as stream:
    document = tomllib.load(stream)
section = document.get("artisan", document)
value = section.get("blender")
if value is not None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Configured blender must be a non-empty path string")
    executable = Path(value).expanduser()
    print(executable if executable.is_absolute() else path.parent / executable)
PY
    )
fi

blender_usable() {
    local output
    output=$("$1" --version 2>/dev/null) || return 1
    [[ ${output%%$'\n'*} == "Blender $BLENDER_VERSION" ]]
}
if [[ -n $blender_bin ]]; then
    blender_bin=$(executable_path "$blender_bin")
    blender_usable "$blender_bin" || die "--blender must run Blender $BLENDER_VERSION"
elif [[ -n ${ARTISAN_BLENDER_BIN:-} ]]; then
    blender_bin=$(executable_path "$ARTISAN_BLENDER_BIN")
    blender_usable "$blender_bin" || die "ARTISAN_BLENDER_BIN must run Blender $BLENDER_VERSION"
elif [[ -n $configured_blender ]]; then
    blender_bin=$(executable_path "$configured_blender")
    blender_usable "$blender_bin" || die "Configured Blender must run version $BLENDER_VERSION"
elif command -v blender >/dev/null && blender_usable "$(command -v blender)"; then
    blender_bin=$(executable_path blender)
else
    blender_bin=$data_dir/tools/blender-$BLENDER_VERSION-linux-x64/blender
    if [[ ! -x $blender_bin ]]; then
        archive=$(fetch_archive "https://download.blender.org/release/Blender5.0/blender-$BLENDER_VERSION-linux-x64.tar.xz" blender-$BLENDER_VERSION-linux-x64.tar.xz)
        tar -xf "$archive" -C "$temp_dir" --no-same-owner
        [[ ! -e $data_dir/tools/blender-$BLENDER_VERSION-linux-x64 ]] || die 'Incomplete Blender installation'
        mv -- "$temp_dir/blender-$BLENDER_VERSION-linux-x64" "$data_dir/tools/"
    fi
    blender_usable "$blender_bin" || die "Blender must run version $BLENDER_VERSION"
fi
log "Using $blender_bin"

node_usable() {
    local version major minor patch
    version=$("$1" --version 2>/dev/null) || return 1
    [[ $version =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || return 1
    major=${BASH_REMATCH[1]}; minor=${BASH_REMATCH[2]}
    (( (major == 20 && minor >= 19) || (major == 22 && minor >= 12) || major >= 24 ))
}
if ((rebuild_viewer)) || [[ ! -f $source_dir/viewer/web/dist/index.html ]]; then
    if [[ -n $node_bin ]]; then
        node_bin=$(executable_path "$node_bin")
        node_usable "$node_bin" || die '--node requires Node 20.19+, 22.12+, or 24+'
    elif command -v node >/dev/null && node_usable "$(command -v node)" && command -v npm >/dev/null; then
        node_bin=$(executable_path node)
    else
        node_bin=$data_dir/tools/node-v$NODE_VERSION-linux-x64/bin/node
        if [[ ! -x $node_bin ]]; then
            archive=$(fetch_archive "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz" node-v$NODE_VERSION-linux-x64.tar.xz)
            tar -xf "$archive" -C "$temp_dir" --no-same-owner
            [[ ! -e $data_dir/tools/node-v$NODE_VERSION-linux-x64 ]] || die 'Incomplete Node installation'
            mv -- "$temp_dir/node-v$NODE_VERSION-linux-x64" "$data_dir/tools/"
        fi
        node_usable "$node_bin" || die 'Downloaded Node could not start'
    fi
    export PATH="$(dirname -- "$node_bin"):$PATH"
    command -v npm >/dev/null || die 'npm must be available beside the selected Node executable'
    log 'Building viewer'
    npm_options=(--no-audit --no-fund)
    ((offline == 0)) || npm_options+=(--offline)
    (cd -- "$source_dir/viewer/web" && npm ci "${npm_options[@]}" && npm run build)
    [[ -f $source_dir/viewer/web/dist/index.html ]] || die 'Viewer build did not produce dist/index.html'
fi

if [[ -d $UV_PROJECT_ENVIRONMENT && ! -f $data_dir/.artisan-managed-environment ]]; then
    die "Refusing to replace an unmarked environment at $UV_PROJECT_ENVIRONMENT"
fi
touch -- "$data_dir/.artisan-managed-environment"
log 'Installing runtime dependencies and USD'
"$uv_bin" sync --project "$source_dir" --frozen --no-dev --no-default-groups --extra usd \
    --python "$python_bin" "${uv_network[@]}"
[[ -x $UV_PROJECT_ENVIRONMENT/bin/python && -x $UV_PROJECT_ENVIRONMENT/bin/artisan ]] || die 'Environment did not produce the Artisan entry point'

log 'Checking native Python modules and headless Blender'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$UV_PROJECT_ENVIRONMENT/bin/python" -I -c 'import cadquery, fcl, mujoco, coacd, numpy, scipy, trimesh; from pxr import Usd; print("Native runtime imports passed")'
"$blender_bin" --background --factory-startup --python-expr 'import bpy; print("Artisan Blender ready")'

init_args=(init --blender "$blender_bin" --workspace "$workspace" --config "$config_path")
[[ -z $credentials ]] || init_args+=(--credentials "$credentials")
"$UV_PROJECT_ENVIRONMENT/bin/python" -I -m artisan "${init_args[@]}"
"$UV_PROJECT_ENVIRONMENT/bin/python" -I -m artisan doctor --config "$config_path"

launcher=$bin_dir/artisan
if [[ -e $launcher || -L $launcher ]]; then
    [[ ! -L $launcher ]] && grep -Eqx '[#:] Artisan managed launcher' "$launcher" || die "Refusing to overwrite an unrelated launcher: $launcher"
fi
{
    printf '#!/usr/bin/env bash\n: Artisan managed launcher\nset -euo pipefail\nunset PYTHONPATH PYTHONHOME\n'
    printf 'if [[ -z ${ARTISAN_CONFIG:-} ]]; then export ARTISAN_CONFIG=%q; fi\n' "$config_path"
    printf 'if [[ -z ${XDG_CACHE_HOME:-} ]]; then export XDG_CACHE_HOME=%q; fi\n' "$data_dir/cache"
    printf 'if [[ -z ${XDG_CONFIG_HOME:-} ]]; then export XDG_CONFIG_HOME=%q; fi\n' "$data_dir/xdg/config"
    printf 'if [[ -z ${XDG_DATA_HOME:-} ]]; then export XDG_DATA_HOME=%q; fi\n' "$data_dir/xdg/data"
    printf 'if [[ -z ${BLENDER_USER_RESOURCES:-} ]]; then export BLENDER_USER_RESOURCES=%q; fi\n' "$data_dir/cache/blender"
    printf 'if [[ -z ${TMPDIR:-} ]]; then export TMPDIR=%q; fi\n' "$data_dir/tmp"
    printf 'exec %q -I -m artisan "$@"\n' "$UV_PROJECT_ENVIRONMENT/bin/python"
} > "$temp_dir/artisan"
chmod 755 "$temp_dir/artisan"
if [[ ! -e $launcher ]] || ! cmp -s "$temp_dir/artisan" "$launcher"; then
    if [[ -e $launcher ]]; then
        backup=$(mktemp "$bin_dir/.artisan.previous.XXXXXXXX")
        cp -p -- "$launcher" "$backup"
        log "Previous launcher preserved at $backup"
    fi
    install -m 755 "$temp_dir/artisan" "$launcher"
fi
log "Installation complete"
printf '%q doctor\n' "$launcher"
case ":$PATH:" in *":$bin_dir:"*) ;; *) printf 'export PATH=%q:"$PATH"\n' "$bin_dir" ;; esac
