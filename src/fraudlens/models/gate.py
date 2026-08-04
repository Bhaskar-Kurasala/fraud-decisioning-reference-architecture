"""The promotion gate: the order the checks run in, and the verdict that follows.

E1's measurement is the specification for this module. A class-rebalanced model
scored AUC 0.9029 against the champion's 0.9045 — a 0.0021 gap no review process
would block — and cost $4.37M/yr more, because its ECE was 0.1389 against 0.0035
and it challenged 52% of traffic instead of 7%. Any gate keyed on discrimination
promotes that model. This one cannot.

Check order, and why it is this order:

1. **Label maturity** — a precondition, not a comparison. Chargebacks arrive
   34-97 days after the transaction, so a window evaluated too early shows a
   fraud rate that is simply wrong and every downstream number inherits the
   error. Evaluating on immature labels is the classic error in fraud ML, and it
   flatters the challenger systematically (fewer observed frauds => the model
   that allows more looks cheaper). If this fails, nothing downstream is
   computed: reporting a cost delta on immature labels would be worse than
   reporting nothing.
2. **Calibration** — cheap, and it is the check that catches the E1 failure.
   Running it before the cost test means the rejection reason recorded in the
   registry names the actual defect ("ECE regressed 40x") rather than a
   downstream symptom ("costs more").
3. **Expected-cost delta** — the decision metric, paired on the same
   transactions with an anytime-valid confidence sequence (see `sequential`).
4. **Segment guard** — last, because it only matters for a challenger that has
   already passed in aggregate. An aggregate win that hides a regression in
   new-account or high-amount traffic is a concentrated harm to an identifiable
   population, which ADR-0002 ranks above the aggregate gain.

Failing is a normal outcome. Every check reports independently and the whole
structure is returned for logging; nothing is suppressed or short-circuited on
first failure except where a failed precondition makes downstream numbers
meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

from fraudlens.models.checks import (
    CheckResult,
    CheckStatus,
    GateInputs,
    GateThresholds,
    check_calibration,
    check_expected_cost,
    check_label_maturity,
    check_segments,
)


@dataclass(frozen=True, slots=True)
class GateDecision:
    promote: bool
    checks: tuple[CheckResult, ...]

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if c.status is CheckStatus.FAILED)

    def summary(self) -> str:
        """One line per check, suitable for a log record or a registry tag."""
        verdict = "PROMOTE" if self.promote else "BLOCK"
        lines = [f"{verdict}"]
        lines.extend(f"  [{c.status.value}] {c.name}: {c.reason}" for c in self.checks)
        return "\n".join(lines)


def evaluate_promotion(
    inputs: GateInputs,
    thresholds: GateThresholds | None = None,
) -> GateDecision:
    """Run the ordered checks and return the full, structured verdict.

    A challenger promotes only if every non-skipped check passes. Skipped checks
    do not count as passes and do not count as failures; they are reported so a
    reviewer can see that, for example, no segment definitions were supplied.
    """
    thresholds = thresholds or GateThresholds()

    maturity = check_label_maturity(inputs, thresholds)
    if maturity.status is CheckStatus.FAILED:
        # Precondition failure. The downstream numbers would be computable but
        # misleading, and a misleading number in an audit log is worse than an
        # absent one.
        skipped = tuple(
            CheckResult(
                name=name,
                status=CheckStatus.SKIPPED,
                reason="not evaluated: labels are not mature enough to compare on",
            )
            for name in ("calibration", "expected_cost", "segment_guard")
        )
        return GateDecision(promote=False, checks=(maturity, *skipped))

    checks = (
        maturity,
        check_calibration(inputs, thresholds),
        check_expected_cost(inputs, thresholds),
        check_segments(inputs, thresholds),
    )
    return GateDecision(
        promote=all(c.status is not CheckStatus.FAILED for c in checks),
        checks=checks,
    )
