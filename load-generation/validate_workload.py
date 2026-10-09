"""Validate workload template files and print summary statistics.

Usage:
    python validate_workload.py <conversation.json>   # one conversation file
    python validate_workload.py <workload_dir>        # a whole workload directory
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from pydantic import ValidationError

from core import Conversation, Role, Workload

# Rough heuristic: ~4 chars per token for English/mixed text.
CHARS_PER_TOKEN = 4.0


def summarize(workload: Workload) -> str:
    lines: list[str] = []
    convs = workload.conversations
    rounds = [c.num_rounds for c in convs]
    single = sum(1 for r in rounds if r == 1)
    manifest = workload.manifest

    lines.append(f"workload_name:        {manifest.workload_name}")
    if manifest.model:
        lines.append(f"model:                {manifest.model}")
    if manifest.tags:
        lines.append(f"tags:                 {', '.join(manifest.tags)}")
    lines.append(f"conversations:        {len(convs)}")
    lines.append(f"  single-round:       {single}")
    lines.append(f"  multi-round:        {len(rounds) - single}")
    lines.append(
        f"rounds/conversation:  min={min(rounds)} "
        f"mean={statistics.fmean(rounds):.1f} max={max(rounds)}"
    )

    input_chars = [
        sum(len(t.value) for t in c.conversations if t.from_ != Role.GPT)
        for c in convs
    ]
    est_tokens = [c / CHARS_PER_TOKEN for c in input_chars]
    lines.append(
        f"est. input tokens:    min={min(est_tokens):.0f} "
        f"mean={statistics.fmean(est_tokens):.0f} max={max(est_tokens):.0f} "
        f"(~{CHARS_PER_TOKEN:.0f} chars/token heuristic)"
    )

    lines.append(f"total weight:         {sum(c.weight for c in convs):.3f}")

    with_offset = sum(1 for c in convs if c.arrival_offset_ms is not None)
    lines.append(f"with arrival_offset_ms:      {with_offset}/{len(convs)}")

    with_expected = sum(1 for c in convs if c.expected_output_tokens is not None)
    lines.append(f"with expected_output_tokens: {with_expected}/{len(convs)}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="A conversation JSON file or a workload directory")
    args = parser.parse_args()

    path = Path(args.path)
    try:
        if path.is_dir():
            workload = Workload.from_directory(path)
            print(f"VALID workload directory: {path}")
            print(summarize(workload))
        else:
            conv = Conversation.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            print(f"VALID conversation file: {path}")
            print(f"id:     {conv.id}")
            print(f"rounds: {conv.num_rounds}")
            print(f"weight: {conv.weight}")
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        print(f"INVALID: {path}\n", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
