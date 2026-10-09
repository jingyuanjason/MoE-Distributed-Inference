"""Recipe-driven prompt sampler.

Reads a YAML recipe describing sampling sources — each source pairs a
template directory with a dedicated sampling function and a relative ratio —
then samples prompts according to the configured mix. Buzzwords are shared
across sources and substituted into the {word} placeholder.

Each template kind has its own module-level sampling function (registered in
SAMPLING_FUNCTIONS), so different template kinds can implement different
sampling logic. See config.example.yaml for the recipe format.

Usage:
    sampler = PromptSampler.from_recipe("config.example.yaml")
    user_queries = sampler.sample()          # list[str]
    user_queries, source = sampler.sample_with_source()
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

BASE_DIR = Path(__file__).parent

if __package__ in (None, ""):
    # Allow running directly as `python core/prompt_sampler.py`.
    sys.path.insert(0, str(BASE_DIR.parent))

from core.workload_schema import Conversation, Role

WORD_PLACEHOLDER = "{word}"
RESPONSE_PLACEHOLDER = "{response}"

# A sampling function picks one template and one buzzword and returns the
# user (human) queries as a list of strings.
SamplingFunction = Callable[[random.Random, list[Conversation], list[str]], list[str]]


def sample_single_round(
    rng: random.Random, templates: list[Conversation], buzzwords: list[str]
) -> list[str]:
    """Single-round template: substitute the buzzword into the one human turn."""
    template = rng.choice(templates)
    buzzword = rng.choice(buzzwords)
    return [
        turn.value.replace(WORD_PLACEHOLDER, buzzword)
        for turn in template.conversations
        if turn.from_ == Role.HUMAN
    ]


def sample_multi_round(
    rng: random.Random, templates: list[Conversation], buzzwords: list[str]
) -> list[str]:
    """Multi-round template: substitute into all human turns, in order.

    Assistant turns hold the {response} placeholder; the caller decides
    whether to replay captured responses or let the engine generate them
    live turn-by-turn. Here we return only the user queries.
    """
    template = rng.choice(templates)
    buzzword = rng.choice(buzzwords)
    return [
        turn.value.replace(WORD_PLACEHOLDER, buzzword)
        for turn in template.conversations
        if turn.from_ == Role.HUMAN
    ]


SAMPLING_FUNCTIONS: dict[str, SamplingFunction] = {
    "single_round": sample_single_round,
    "multi_round": sample_multi_round,
}


@dataclass
class SamplingSource:
    """One entry of the recipe: templates + sampling function + ratio."""

    name: str
    type: str
    templates: list[Conversation]
    ratio: float
    fn: SamplingFunction



class PromptSampler:
    """Samples prompts from multiple sources according to a recipe's ratios."""

    def __init__(
        self,
        buzzwords: list[str],
        sources: list[SamplingSource],
        seed: int | None = None,
    ) -> None:
        if not buzzwords:
            raise ValueError("buzzword list is empty")
        if not sources:
            raise ValueError("recipe has no sources")
        if any(s.ratio <= 0 for s in sources):
            raise ValueError("source ratios must be positive")
        self.buzzwords = buzzwords
        self.sources = sources
        self._rng = random.Random(seed)

    @classmethod
    def from_recipe(cls, recipe_path: str | Path) -> "PromptSampler":
        """Build a sampler from a YAML recipe file.

        Relative paths inside the recipe are resolved against the recipe
        file's directory.
        """
        recipe_path = Path(recipe_path)
        recipe_dir = recipe_path.parent
        with open(recipe_path, encoding="utf-8") as f:
            recipe = yaml.safe_load(f)

        buzzwords_file = recipe_dir / recipe.get(
            "buzzwords_file", "templates/buzzwords.txt"
        )
        buzzwords = [
            line.strip()
            for line in buzzwords_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        sources: list[SamplingSource] = []
        for entry in recipe["sources"]:
            type_ = entry["type"]
            if type_ not in SAMPLING_FUNCTIONS:
                raise ValueError(
                    f"unknown source type {type_!r}; "
                    f"known types: {sorted(SAMPLING_FUNCTIONS)}"
                )
            templates_dir = recipe_dir / entry["templates_dir"]
            templates = [
                Conversation.model_validate(
                    json.loads(file.read_text(encoding="utf-8"))
                )
                for file in sorted(templates_dir.glob("*.json"))
                if file.name != "workload.json"
            ]
            if not templates:
                raise ValueError(f"no template files found in {templates_dir}")
            sources.append(
                SamplingSource(
                    name=entry.get("name", type_),
                    type=type_,
                    templates=templates,
                    ratio=float(entry["ratio"]),
                    fn=SAMPLING_FUNCTIONS[type_],
                )
            )
        return cls(buzzwords=buzzwords, sources=sources, seed=recipe.get("seed"))

    def _pick_source(self) -> SamplingSource:
        return self._rng.choices(
            self.sources, weights=[s.ratio for s in self.sources], k=1
        )[0]

    def sample(self) -> list[str]:
        """Sample one prompt: pick a source by ratio, then run its function.

        Returns the user queries as a list — one entry for single-round
        sources, one per human turn for multi-round sources.
        """
        source = self._pick_source()
        return source.fn(self._rng, source.templates, self.buzzwords)

    def sample_with_source(self) -> tuple[list[str], str]:
        """Like sample(), but also returns the source name (for metrics)."""
        source = self._pick_source()
        return source.fn(self._rng, source.templates, self.buzzwords), source.name


if __name__ == "__main__":
    from collections import Counter

    sampler = PromptSampler.from_recipe(BASE_DIR.parent / "config.example.yaml")
    print(f"buzzwords: {len(sampler.buzzwords)}")
    for s in sampler.sources:
        print(f"source {s.name!r}: {len(s.templates)} templates, ratio {s.ratio}")

    print("\n--- samples ---")
    for _ in range(3):
        print(sampler.sample())

    counts = Counter(sampler.sample_with_source()[1] for _ in range(10_000))
    total = sum(counts.values())
    print("\n--- source distribution over 10k samples ---")
    for name, n in counts.items():
        print(f"{name}: {n / total:.1%}")
