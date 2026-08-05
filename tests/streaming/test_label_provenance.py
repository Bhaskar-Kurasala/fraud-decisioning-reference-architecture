"""Label provenance: human adjudications stay out of training unless explicitly let in.

The tests here are the contract E12c exists to enforce. Three claims, each
asserted structurally rather than by convention:

1. **A human label cannot enter the training/promotion path by accident.**
   The default label set is chargeback-only; human labels require an explicit
   ``include_human=True`` and are tagged with their origin even then.
2. **Chargeback wins on disagreement.** The outcome is ground truth; the
   opinion was an early guess. The disagreement itself is recorded because it
   is the control-chart signal.
3. **The prevented-loss case does not silently become a fraud label.** A human
   "fraud" call with no chargeback is origin=HUMAN, excluded from training by
   default, and the reconciliation does not fabricate an outcome from an
   opinion.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import Engine, select, text
from sqlalchemy.exc import IntegrityError

from fraudlens.streaming.labels import (
    HumanAdjudication,
    LabelOrigin,
    RevealedLabel,
    effective_labels,
    read_adjudications,
    reconcile,
    record_adjudication,
)
from fraudlens.streaming.schema import revealed_labels

from .conftest import EPOCH

STAMP = EPOCH + dt.timedelta(days=10)
DAY_50 = EPOCH + dt.timedelta(days=50)
DAY_100 = EPOCH + dt.timedelta(days=100)


def _chargeback(tid: int, is_fraud: bool, revealed: dt.datetime = DAY_100) -> RevealedLabel:
    return RevealedLabel(
        transaction_id=tid,
        is_fraud=is_fraud,
        transaction_at=EPOCH,
        revealed_at=revealed,
        recorded_at=revealed,
        dispute_lag_days=(revealed - EPOCH).days,
        label_source="chargeback",
    )


def _human(
    tid: int, is_fraud: bool, at: dt.datetime = DAY_50, who: str = "analyst-1"
) -> HumanAdjudication:
    return HumanAdjudication(
        transaction_id=tid,
        is_fraud=is_fraud,
        adjudicated_at=at,
        adjudicator=who,
    )


# --- recording and reading -------------------------------------------------


def test_a_new_adjudication_is_written_and_returns_true(engine: Engine) -> None:
    assert record_adjudication(engine, _human(1, True)) is True
    assert read_adjudications(engine, DAY_100) == [_human(1, True)]


def test_a_duplicate_adjudication_is_a_noop_not_a_correction(engine: Engine) -> None:
    """Same person, same transaction: the second write is silently ignored,
    not an update. A correction is a different adjudicator (second opinion)."""
    record_adjudication(engine, _human(1, True))
    # Same person, opposite call — still ignored, because append-only.
    assert record_adjudication(engine, _human(1, False)) is False
    rows = read_adjudications(engine, DAY_100)
    assert len(rows) == 1
    assert rows[0].is_fraud is True  # the original call stands


def test_a_second_adjudicator_is_a_separate_row(engine: Engine) -> None:
    record_adjudication(engine, _human(1, True, who="analyst-1"))
    assert record_adjudication(engine, _human(1, False, who="analyst-2")) is True
    rows = read_adjudications(engine, DAY_100)
    assert len(rows) == 2


def test_adjudications_are_filtered_by_as_of(engine: Engine) -> None:
    record_adjudication(engine, _human(1, True, at=DAY_50))
    record_adjudication(engine, _human(2, False, at=DAY_100))
    assert len(read_adjudications(engine, DAY_50)) == 1
    assert len(read_adjudications(engine, DAY_100)) == 2


def test_a_naive_adjudication_timestamp_is_rejected(engine: Engine) -> None:
    bad = HumanAdjudication(
        transaction_id=1,
        is_fraud=True,
        adjudicated_at=dt.datetime(2026, 1, 1),  # noqa: DTZ001
        adjudicator="analyst-1",
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        record_adjudication(engine, bad)


def test_human_adjudications_table_is_append_only(engine: Engine) -> None:
    """An adjudication is a historical fact. 'This person said this' is not editable."""
    record_adjudication(engine, _human(1, True))
    with pytest.raises(IntegrityError, match="append-only"), engine.begin() as conn:
        conn.execute(text("UPDATE human_adjudications SET is_fraud = 0"))


# --- reconciliation --------------------------------------------------------


def test_chargeback_wins_when_both_exist() -> None:
    """The outcome is ground truth; the opinion was an early guess."""
    effective = reconcile(_human(1, is_fraud=False), _chargeback(1, is_fraud=True))
    assert effective is not None
    assert effective.is_fraud is True
    assert effective.origin is LabelOrigin.CHARGEBACK


def test_a_disagreement_is_flagged_not_hidden() -> None:
    """The human-call-was-wrong event is the control-chart signal (E12b)."""
    effective = reconcile(_human(1, is_fraud=True), _chargeback(1, is_fraud=False))
    assert effective is not None
    assert effective.disagrees_with_human is True


def test_an_agreement_does_not_flag_a_disagreement() -> None:
    effective = reconcile(_human(1, is_fraud=True), _chargeback(1, is_fraud=True))
    assert effective is not None
    assert effective.disagrees_with_human is False


def test_human_only_label_carries_the_human_origin() -> None:
    effective = reconcile(_human(1, is_fraud=True), None)
    assert effective is not None
    assert effective.origin is LabelOrigin.HUMAN
    assert effective.disagrees_with_human is False


def test_no_labels_yields_no_effective_label() -> None:
    assert reconcile(None, None) is None


# --- the default path is structurally safe ---------------------------------


def test_effective_labels_defaults_to_chargeback_only() -> None:
    """The training and promotion path. Human labels are invisible here unless
    the caller explicitly opts in, and even then they are tagged."""
    chargebacks = [_chargeback(1, True), _chargeback(2, False)]
    humans = [_human(1, False), _human(3, True)]  # tid 3 has no chargeback

    labels = effective_labels(chargebacks, humans=humans)

    assert len(labels) == 2
    assert all(lb.origin is LabelOrigin.CHARGEBACK for lb in labels)
    tids = {lb.transaction_id for lb in labels}
    assert tids == {1, 2}  # tid 3 (human-only) excluded


def test_the_default_path_cannot_even_accidentally_include_a_human(engine: Engine) -> None:
    """The promotion gate reads revealed_labels directly (monitoring.report joins
    on it). A human adjudication in a separate table is structurally invisible
    to that query, not filtered out by convention."""
    record_adjudication(engine, _human(1, True))

    with engine.connect() as conn:
        count = conn.execute(
            select(revealed_labels).where(revealed_labels.c.transaction_id == 1)
        ).fetchall()

    assert count == []  # the human label is not in revealed_labels


def test_opt_in_includes_human_only_labels_tagged_with_origin() -> None:
    chargebacks = [_chargeback(1, True)]
    humans = [_human(1, False), _human(2, True)]

    labels = effective_labels(chargebacks, humans=humans, include_human=True)

    assert len(labels) == 2
    by_tid = {lb.transaction_id: lb for lb in labels}
    # tid 1: both exist, chargeback wins, disagreement flagged.
    assert by_tid[1].origin is LabelOrigin.CHARGEBACK
    assert by_tid[1].disagrees_with_human is True
    # tid 2: human only.
    assert by_tid[2].origin is LabelOrigin.HUMAN


def test_second_opinion_uses_the_latest_adjudication() -> None:
    """When two analysts adjudicate the same transaction, the later call wins
    for the human label — it is the more considered opinion."""
    humans = [
        _human(1, True, at=DAY_50, who="analyst-1"),
        _human(1, False, at=DAY_100, who="analyst-2"),
    ]
    labels = effective_labels([], humans=humans, include_human=True)
    assert len(labels) == 1
    assert labels[0].is_fraud is False  # the second, later call


# --- the prevented-loss case -----------------------------------------------


def test_a_human_fraud_call_without_a_chargeback_is_human_not_chargeback() -> None:
    """The ambiguity: the analyst may have been right (prevented loss) or wrong
    (false positive). The reconciliation does not resolve this by fabricating
    an outcome — the label stays origin=HUMAN, and training excludes it."""
    effective = reconcile(_human(1, is_fraud=True), None)
    assert effective is not None
    assert effective.origin is LabelOrigin.HUMAN
    assert effective.is_fraud is True  # the opinion is preserved
    # But it does not enter the default training set:
    labels = effective_labels([], humans=[_human(1, True)])
    assert labels == []
