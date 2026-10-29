#!/usr/bin/env bash
# Install the vendored sources into the container's current Python environment.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-$(command -v python || command -v python3)}
source "$ROOT/env.sh"
exec 9>"$ROOT/.install.lock"
flock 9
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
cat "$ROOT/constraints.txt" > "$work_dir/constraints.txt"
"$PYTHON" - >> "$work_dir/constraints.txt" <<'PY'
import importlib.metadata as metadata
import sys
if sys.version_info[:2] not in ((3, 11), (3, 12)):
    raise SystemExit('Use Python 3.11 or 3.12; local validation used Python 3.11.')
for name in ('torch', 'torch-npu', 'torchvision', 'torchaudio'):
    try:
        version = metadata.distribution(name).version
    except metadata.PackageNotFoundError:
        if name in ('torch', 'torch-npu'):
            raise SystemExit(f'{name} must already be installed in the base container')
        continue
    if name in ('torch', 'torch-npu') and version.split('.')[:2] != ['2', '10']:
        raise SystemExit(f'Use a Torch/TorchNPU 2.10 base container; found {name}={version}')
    print(f'{name}=={version}')
PY
export PIP_CONSTRAINT="$work_dir/constraints.txt"
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}
export PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL:-https://mirrors.huaweicloud.com/ascend/repos/pypi}
"$PYTHON" -m pip install 'setuptools>=80' wheel
# Some old optional decord wheels advertise CPython 3.6-only internal tags.
if "$PYTHON" - <<'PY'
import importlib.metadata as metadata
from packaging.tags import parse_tag, sys_tags
try:
    wheel = metadata.distribution('decord').read_text('WHEEL') or ''
except metadata.PackageNotFoundError:
    raise SystemExit(1)
tags = set()
for line in wheel.splitlines():
    if line.startswith('Tag: '):
        tags.update(parse_tag(line[5:]))
raise SystemExit(0 if tags and not tags.intersection(sys_tags()) else 1)
PY
then
    echo 'Removing incompatible optional decord wheel; video input is outside this validation.'
    "$PYTHON" -m pip uninstall -y decord
fi
"$PYTHON" -m pip install -r "$ROOT/requirements.txt"
for repo in Megatron-LM MegatronAdaptor TransformerEngineNPU mcore-bridge ms-swift; do
    echo "Installing $repo from $ROOT/$repo"
    (cd "$ROOT/$repo" && "$PYTHON" -m pip install --no-build-isolation -e .)
done
"$PYTHON" -m pip check
"$PYTHON" "$ROOT/verify_install.py"
printf '\nInstalled. In each training shell run:\nsource %q\n' "$ROOT/env.sh"
