#!/usr/bin/env python3
"""Step ⑦ PATCH — ast.parse validator. LLM generates patch, this validates.

Loop: LLM generates → ast.parse → fail? feed error back → retry → pass → done.
"""

import ast
import sys
from pathlib import Path


def validate_patch_code(filepath: str) -> dict:
    """Run ast.parse on a patch file. Returns {passed, error}."""
    try:
        content = Path(filepath).read_text(encoding="utf-8")
        ast.parse(content)
        return {"passed": True, "error": None}
    except SyntaxError as e:
        return {"passed": False, "error": f"line {e.lineno}: {e.msg}"}
    except FileNotFoundError:
        return {"passed": False, "error": f"file not found: {filepath}"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: nightly_patch.py <patch_file.py>")
        sys.exit(1)
    result = validate_patch_code(sys.argv[1])
    if result["passed"]:
        print("AST OK")
        sys.exit(0)
    else:
        print(f"AST FAIL: {result['error']}")
        sys.exit(1)
