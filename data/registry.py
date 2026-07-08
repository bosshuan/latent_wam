"""Embodiment / action-schema registry.

Two roles (CLAUDE.md §4):
  * **OXE subset = expectation assertions.** For datasets whose dims we already
    know, register the expected ``(state_dim, action_dim)`` so a mismatch at
    scan time fails early instead of silently mis-padding.
  * **Scanned datasets** (AgiBot / InternData-A1 / RoboCOIN) are *not*
    pre-registered; the schema scanner reads their dims from ``info.json`` and
    allocates an ``action_schema_id``. RoboTwin 2.0 gets a reserved
    ``NEW_EMBODIMENT`` slot but is **not trained** in Stage A.

Embodiment vs action-schema:
  * ``embodiment_id`` selects the CategorySpecific adapter weights (per-robot).
  * ``action_schema_id`` groups samples with the same padded action layout +
    semantics so counterfactual permutation / collate batching is valid.
Several datasets can share one ``action_schema_id`` while differing in
``embodiment_id`` and vice-versa, so they are tracked independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Name of the RoboTwin 2.0 benchmark embodiment (slot kept, never trained in
# Stage A). It is allocated a *trailing* id on demand — NOT id 0 — so it can
# never collide with the default/uninitialized ``embodiment_id=0`` that real
# trained embodiments use (otherwise robot data could silently route through the
# untrained NEW_EMBODIMENT CategorySpecific weights without erroring).
NEW_EMBODIMENT = "NEW_EMBODIMENT"

# Sentinel for "embodiment not specified" (e.g. actionless video, which never
# selects an action/state CategorySpecific adapter). Distinct from every real
# id and from NEW_EMBODIMENT. Adapters must refuse to index a -1.
INVALID_EMBODIMENT_ID = -1


@dataclass(frozen=True)
class ExpectedSchema:
    """An assertion target for a known (OXE-subset) dataset."""

    dataset_id: str
    embodiment: str
    state_dim: int
    action_dim: int
    note: str = ""


@dataclass
class EmbodimentRegistry:
    """Holds expected schemas + assigns integer ids deterministically.

    Real embodiments are assigned ids in first-seen order starting at 0.
    ``NEW_EMBODIMENT`` is never pre-seeded; it is allocated a trailing id only
    when explicitly requested via ``new_embodiment_id()`` so id 0 always belongs
    to a real trained embodiment.
    """

    expected: dict[str, ExpectedSchema] = field(default_factory=dict)
    _embodiment_ids: dict[str, int] = field(default_factory=dict)
    _schema_ids: dict[tuple, int] = field(default_factory=dict)

    # --- registration of known expectations -----------------------------
    def register_expected(self, schema: ExpectedSchema) -> None:
        self.expected[schema.dataset_id] = schema

    def expected_for(self, dataset_id: str) -> Optional[ExpectedSchema]:
        return self.expected.get(dataset_id)

    # --- id allocation --------------------------------------------------
    def embodiment_id(self, embodiment: str) -> int:
        """Allocate/return the id for a *real* embodiment (first-seen, from 0).

        Refuses to hand out an id for ``NEW_EMBODIMENT`` through this path — use
        ``new_embodiment_id()`` so the untrained benchmark slot is explicit and
        always trailing.
        """
        if embodiment == NEW_EMBODIMENT:
            raise ValueError(
                "use new_embodiment_id() for the untrained NEW_EMBODIMENT slot; "
                "it must not occupy a real-embodiment id (e.g. 0)."
            )
        if embodiment not in self._embodiment_ids:
            self._embodiment_ids[embodiment] = len(self._embodiment_ids)
        return self._embodiment_ids[embodiment]

    def new_embodiment_id(self) -> int:
        """Allocate the trailing reserved id for the RoboTwin benchmark slot."""
        if NEW_EMBODIMENT not in self._embodiment_ids:
            self._embodiment_ids[NEW_EMBODIMENT] = len(self._embodiment_ids)
        return self._embodiment_ids[NEW_EMBODIMENT]

    @property
    def reserved_new_embodiment_id(self) -> Optional[int]:
        return self._embodiment_ids.get(NEW_EMBODIMENT)

    def action_schema_id(self, action_dim: int, state_dim: int, semantics: str) -> int:
        key = (int(action_dim), int(state_dim), str(semantics))
        if key not in self._schema_ids:
            self._schema_ids[key] = len(self._schema_ids)
        return self._schema_ids[key]

    def is_trainable(self, embodiment: str) -> bool:
        return embodiment != NEW_EMBODIMENT

    def assert_trainable_batch(self, embodiment_ids) -> None:
        """Fail loud if a *training* batch references untrained/invalid ids.

        Guards against routing real robot data through the untrained
        NEW_EMBODIMENT adapter, or leaving an embodiment_id unset (-1).
        ``embodiment_ids`` is any iterable / 1-D tensor of ints.
        """
        ids = [int(x) for x in embodiment_ids]
        if any(i == INVALID_EMBODIMENT_ID for i in ids):
            raise ValueError(
                f"training batch contains INVALID_EMBODIMENT_ID "
                f"({INVALID_EMBODIMENT_ID}); every trained sample needs a real "
                "embodiment id."
            )
        reserved = self.reserved_new_embodiment_id
        if reserved is not None and any(i == reserved for i in ids):
            raise ValueError(
                f"training batch contains NEW_EMBODIMENT id ({reserved}); the "
                "RoboTwin benchmark slot is not trained in Stage A."
            )

    # --- assertion against a scanned dataset ----------------------------
    def assert_matches_expected(
        self, dataset_id: str, state_dim: int, action_dim: int
    ) -> None:
        exp = self.expected_for(dataset_id)
        if exp is None:
            return  # not an asserted (OXE) dataset; scanner allocates fresh
        if (state_dim, action_dim) != (exp.state_dim, exp.action_dim):
            raise ValueError(
                f"schema mismatch for {dataset_id}: scanned "
                f"(state={state_dim}, action={action_dim}) != expected "
                f"(state={exp.state_dim}, action={exp.action_dim}). {exp.note}"
            )


def default_registry() -> EmbodimentRegistry:
    """Registry seeded with placeholder OXE expectations.

    NOTE: the concrete OXE-subset dims are filled from the real ``info.json``
    files on the server during M1 bring-up; these entries are intentionally
    sparse here so the assertion *mechanism* is testable without committing to
    numbers we have not yet read from data.
    """
    return EmbodimentRegistry()
