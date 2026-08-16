"""Dictionary/gazetteer entity extraction over article text.

Not a trained NER model -- word-boundary, case-insensitive matching against
a curated gazetteer of real, ontology-grounded entities (see
gazetteer.json's _provenance). This is a deliberate scope choice: a trained
biomedical NER model (scispaCy etc.) would catch entities outside the
gazetteer, but needs a model download/dependency this repo doesn't already
carry, and "matches a curated real-ontology list" is a more honest and
inspectable v0.8 slice than a model whose recall/precision on this corpus
was never measured. Aliases are matched longest-first so e.g. "type 2
diabetes" doesn't get shadowed by a shorter unrelated alias.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

GAZETTEER_PATH = Path(__file__).parent / "gazetteer.json"


@dataclass(frozen=True)
class GazetteerEntity:
    name: str
    entity_type: str
    external_source: str | None
    external_id: str | None
    aliases: tuple[str, ...]


@dataclass
class EntityMatch:
    entity: GazetteerEntity
    mention_text: str


def _compile_pattern(alias: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def load_gazetteer() -> list[GazetteerEntity]:
    data = json.loads(GAZETTEER_PATH.read_text())
    return [
        GazetteerEntity(
            name=row["name"],
            entity_type=row["type"],
            external_source=row.get("external_source"),
            external_id=row.get("external_id"),
            aliases=tuple(row["aliases"]),
        )
        for row in data["entities"]
    ]


@lru_cache(maxsize=1)
def _alias_index() -> list[tuple[re.Pattern, GazetteerEntity]]:
    """(compiled alias pattern, entity), longest alias first so a longer,
    more specific alias matches before a shorter substring of it would.
    """
    pairs = [
        (alias, entity)
        for entity in load_gazetteer()
        for alias in entity.aliases
    ]
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return [(_compile_pattern(alias), entity) for alias, entity in pairs]


def extract_entities(text: str) -> list[EntityMatch]:
    """Return one match per distinct entity found in `text` (first surface
    form seen), not one per occurrence -- callers that want mention counts
    should re-derive them from the raw text themselves.
    """
    if not text:
        return []

    seen_entities: dict[str, EntityMatch] = {}
    covered_spans: list[tuple[int, int]] = []

    for pattern, entity in _alias_index():
        if entity.name in seen_entities:
            continue
        match = pattern.search(text)
        if not match:
            continue
        span = match.span()
        # Skip a match that overlaps a longer alias already claimed (e.g.
        # "lymphoma" inside an already-matched "non-Hodgkin lymphoma").
        if any(span[0] < end and start < span[1] for start, end in covered_spans):
            continue
        seen_entities[entity.name] = EntityMatch(entity=entity, mention_text=match.group(0))
        covered_spans.append(span)

    return list(seen_entities.values())
