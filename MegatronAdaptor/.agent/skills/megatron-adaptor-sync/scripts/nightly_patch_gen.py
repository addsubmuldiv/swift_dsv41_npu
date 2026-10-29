#!/usr/bin/env python3
"""Step ⑦ PATCH (generate) — Build a repair prompt from DIAG output + MA source.

Reads the LLM diagnosis (root_cause / affected_patch_file / fix_summary),
loads the current MA patch source, and constructs a constrained LLM prompt
asking for the corrected code.

Does NOT invoke the LLM itself — the orchestrator feeds the prompt to the
LLM and pipes the response into nightly_patch.py for AST validation.
"""

import argparse
import json
from pathlib import Path


def read_patch_source(adaptor_path: str, affected_file: str) -> dict:
    """Read the affected MA patch source file."""
    if not affected_file or not adaptor_path:
        return {"path": "", "content": "", "exists": False}

    file_path = Path(adaptor_path) / affected_file
    if not file_path.exists():
        return {"path": str(file_path), "content": "", "exists": False}

    try:
        content = file_path.read_text()
    except Exception:
        return {"path": str(file_path), "content": "", "exists": False}

    return {"path": str(file_path), "content": content, "exists": True}


def build_patch_prompt(diag: dict, source_info: dict) -> str:
    """Build a repair prompt with DIAG context and current source code."""
    parts = []

    parts.append(
        "You are fixing a bug in a MegatronAdaptor patch file. "
        "The adaptor monkey-patches Megatron-LM functions to run on Ascend NPU "
        "with TransformerEngine."
    )
    parts.append("")

    parts.append("## Diagnosis")
    parts.append("")
    parts.append(f"- Root cause: {diag.get('root_cause', 'N/A')}")
    parts.append(f"- Affected file: {diag.get('affected_patch_file', 'N/A')}")
    parts.append(f"- Fix strategy: {diag.get('fix_summary', 'N/A')}")
    parts.append(f"- Megatron file involved: {diag.get('affected_megatron_file', 'N/A')}")
    parts.append("")

    if source_info.get("exists"):
        parts.append("## Current Source Code (to be fixed)")
        parts.append("")
        parts.append(f"File: `{source_info['path']}`")
        parts.append("```python")
        parts.append(source_info["content"])
        parts.append("```")
        parts.append("")
    else:
        parts.append("## Current Source Code")
        parts.append("")
        parts.append(f"(Could not read: {source_info.get('path', 'unknown')})")
        parts.append("")

    parts.append("## Task")
    parts.append("")
    parts.append("1. Fix the bug described in the diagnosis above.")
    parts.append("2. Make the minimal change necessary — do NOT refactor unrelated code.")
    parts.append("3. Preserve the existing code style, imports, and indentation.")
    parts.append(
        "4. Keep code comments concise and clear: one or two sentences stating what the code does and why. "
        "Do NOT write historical context, cross-file archaeology, or upstream commit provenance into "
        "comments/docstrings — put that in the Issue comment or DIAGNOSIS report instead. "
        "Prefer clear naming over lengthy docstrings."
    )
    parts.append("5. Output ONLY the corrected version of the entire file.")
    parts.append("")
    parts.append("## Output Format")
    parts.append("")
    parts.append("Output the complete corrected file:")
    parts.append("```python")
    parts.append("# ... corrected code here ...")
    parts.append("```")
    parts.append("")
    parts.append("Do NOT include explanations, commentary, or markdown outside the code block.")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Build a repair prompt from DIAG output and MA patch source")
    parser.add_argument("--diag-output", required=True, help="LLM diagnosis JSON (output of DIAG stage after LLM call)")
    parser.add_argument(
        "--adaptor-path", required=True, help="Path to MegatronAdaptor clone (to read affected patch source)"
    )
    parser.add_argument("--output-prompt", required=True, help="Output patch_prompt.txt path (feed this to LLM)")
    args = parser.parse_args()

    with open(args.diag_output, encoding="utf-8") as f:
        diag = json.load(f)

    affected_file = diag.get("affected_patch_file", "")
    source_info = read_patch_source(args.adaptor_path, affected_file)

    prompt = build_patch_prompt(diag, source_info)

    output_path = Path(args.output_prompt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")

    print(f"OK: Patch prompt written to {args.output_prompt}")
    print(f"  Affected file: {affected_file}")
    print(f"  Source loaded: {source_info['exists']}")
    print(f"  Prompt length: {len(prompt)} chars")


if __name__ == "__main__":
    main()
