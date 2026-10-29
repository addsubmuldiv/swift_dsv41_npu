"""Keep both architecture pins consistent with the published Ascend wheel metadata."""
import json
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arch', ['x86_64', 'aarch64'])
def test_triton_constraints_match_wheel_metadata(arch):
    records = json.loads((ROOT / 'tests/fixtures/triton_ascend_3.2.2_metadata.json').read_text())
    requirements = [Requirement(line) for line in (ROOT / 'constraints.txt').read_text().splitlines()
                    if line and not line.startswith('#')]
    active = {}
    for requirement in requirements:
        if requirement.marker is None or requirement.marker.evaluate({'platform_machine': arch}):
            assert requirement.name not in active, requirement.name
            active[requirement.name] = requirement
    for entry in records[arch]['requires_dist']:
        dependency = Requirement(entry)
        if dependency.name not in ('triton', 'pytest'):
            continue
        if dependency.marker is not None and not dependency.marker.evaluate({'platform_machine': arch, 'extra': ''}):
            continue
        selected = next(iter(active[dependency.name].specifier)).version
        assert selected in dependency.specifier, (arch, entry, selected)
