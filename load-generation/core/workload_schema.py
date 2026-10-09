"""Workload template schema for the load generator.

One JSON file = one conversation (single-round or multi-round; single-round
is simply a conversation with exactly one human turn). A workload is composed
at load time from a directory of conversation files, plus an optional
workload manifest (`workload.json`) carrying shared metadata/defaults.

Directory layout:
    my_workload/
    ├── workload.json        # optional manifest (WorkloadManifest)
    ├── chat-0001.json       # one Conversation per file
    ├── chat-0002.json
    └── ...
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Role(StrEnum):
    SYSTEM = "system"
    HUMAN = "human"
    GPT = "gpt"


class SamplingParams(BaseModel):
    """OpenAI/vLLM-compatible sampling parameters."""

    model_config = ConfigDict(extra="allow")

    temperature: Optional[float] = Field(default=None, ge=0.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=-1)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    seed: Optional[int] = None


class Turn(BaseModel):
    """One conversation turn, ShareGPT-style."""

    from_: Role = Field(alias="from")
    value: str

    model_config = ConfigDict(populate_by_name=True)


class Conversation(BaseModel):
    """One replayable request/session = one JSON file.

    A single-round conversation has exactly one human turn; a multi-round
    conversation has alternating human/gpt turns ending with a final human
    turn.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    conversations: list[Turn] = Field(min_length=1)
    weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Relative sampling weight when composing workload mixes "
        "(e.g. the 45/30/20/5 mix in plan.md).",
    )
    arrival_offset_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Replay delay from workload start; if absent the generator "
        "falls back to rate-based arrivals.",
    )
    sampling_params: Optional[SamplingParams] = Field(
        default=None, description="Overrides manifest-level default_sampling_params."
    )
    expected_output_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Observed output length of the final turn; lets the generator "
        "set max_tokens for closed-loop output-length control.",
    )

    @model_validator(mode="after")
    def check_conversation_structure(self) -> "Conversation":
        turns = self.conversations
        start = 0
        if turns[0].from_ == Role.SYSTEM:
            start = 1
        body = turns[start:]
        if not body:
            raise ValueError(f"conversation {self.id!r}: no human turn")
        for i, turn in enumerate(body):
            expected = Role.HUMAN if i % 2 == 0 else Role.GPT
            if turn.from_ != expected:
                raise ValueError(
                    f"conversation {self.id!r}: turn {start + i} must be "
                    f"{expected.value!r}, got {turn.from_.value!r} "
                    "(turns must alternate human/gpt after an optional system turn)"
                )
        if body[-1].from_ != Role.HUMAN:
            raise ValueError(
                f"conversation {self.id!r}: must end with a human turn "
                "(the final human turn is what gets served)"
            )
        return self

    @property
    def num_rounds(self) -> int:
        return sum(1 for t in self.conversations if t.from_ == Role.HUMAN)


class WorkloadManifest(BaseModel):
    """Optional shared metadata/defaults for a directory of conversations."""

    model_config = ConfigDict(extra="allow")

    workload_name: str
    model: Optional[str] = None
    collected_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    default_sampling_params: Optional[SamplingParams] = None


MANIFEST_FILENAME = "workload.json"


class Workload(BaseModel):
    """A manifest plus its conversations, composed from a directory."""

    manifest: WorkloadManifest
    conversations: list[Conversation] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique_ids(self) -> "Workload":
        ids = [c.id for c in self.conversations]
        if len(ids) != len(set(ids)):
            raise ValueError("conversation ids must be unique within a workload")
        return self

    @classmethod
    def from_directory(cls, path: str | Path) -> "Workload":
        """Load all conversation JSON files in a directory (+ optional manifest)."""
        directory = Path(path)
        if not directory.is_dir():
            raise ValueError(f"not a directory: {directory}")

        manifest_path = directory / MANIFEST_FILENAME
        if manifest_path.exists():
            manifest = WorkloadManifest.model_validate(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        else:
            manifest = WorkloadManifest(workload_name=directory.name)

        conversations = []
        for file in sorted(directory.glob("*.json")):
            if file.name == MANIFEST_FILENAME:
                continue
            conversations.append(
                Conversation.model_validate(json.loads(file.read_text(encoding="utf-8")))
            )
        if not conversations:
            raise ValueError(f"no conversation files found in {directory}")
        return cls(manifest=manifest, conversations=conversations)
