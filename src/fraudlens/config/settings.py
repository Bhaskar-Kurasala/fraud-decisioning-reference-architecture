"""Business constants the decision economics are computed from.

Every default here is the exact value `fraud-decisioning-findings.md` was computed
with (root `config.py`, the provenance copy the research scripts still import).
Changing one changes money: the golden-value tests in `tests/golden/` fail if any
published figure moves, which is the intended tripwire.

Values are env-overridable (`FRAUDLENS_COGS=0.68`) so an operator can run a
sensitivity analysis without a code change, but the defaults are the audited ones.
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Tenure buckets. Right-closed: bucket i holds edges[i] < D1 <= edges[i+1], which is
# `pd.cut` semantics and therefore what the published numbers were bucketed with.
# D1 = 0 (first ever transaction on the card) lands in "new(0d)" via the -1 lower edge.
TENURE_EDGES: tuple[float, ...] = (-1.0, 0.0, 7.0, 30.0, 90.0, 180.0, 400.0, 1e9)
TENURE_LABELS: tuple[str, ...] = (
    "new(0d)",
    "1-7d",
    "8-30d",
    "31-90d",
    "91-180d",
    "181-400d",
    "400d+",
)

# Sentinel for a transaction whose D1 is missing or outside the bucket range. 41 of the
# 92,427 test-window rows are in this state, so it is a real production case, not a
# defensive branch. See `economics.costs.relationship_cost` for how it is priced.
UNKNOWN_TENURE = "unknown"


class BusinessConstants(BaseSettings):
    """The merchant's unit economics and intervention parameters.

    Frozen: a cost function whose constants can be mutated mid-process cannot be
    reasoned about from a decision record after the fact, and every decline here is
    subject to dispute.
    """

    model_config = SettingsConfigDict(
        env_prefix="FRAUDLENS_", frozen=True, extra="forbid", case_sensitive=False
    )

    # --- merchant unit economics -------------------------------------------------
    # BUSINESS ASSUMPTION: a fraud loss costs cost-of-goods, not retail price. Review
    # when the product mix (and therefore the blended COGS) changes.
    cogs: float = Field(default=0.70, gt=0.0, lt=1.0)
    margin: float = Field(default=0.30, gt=0.0, lt=1.0)

    # --- false-negative (undetected fraud) ---------------------------------------
    cb_fee: float = Field(default=25.0, ge=0.0)  # network + acquirer chargeback fee
    ops_dispute: float = Field(default=12.0, ge=0.0)  # internal dispute handling

    # --- intervention parameters -------------------------------------------------
    # BUSINESS ASSUMPTION (all five): vendor/industry figures, not measured on this
    # dataset. No transaction in IEEE-CIS was ever actually challenged or reviewed, so
    # the challenge and review arms of the cost model are counterfactual. f_pass alone
    # drives the P1->P4 gap and the findings' sensitivity grid flips the review
    # conclusion above f_pass ~= 0.20.
    f_pass: float = Field(default=0.11, ge=0.0, le=1.0)  # fraud that survives a step-up
    a_abandon: float = Field(default=0.07, ge=0.0, le=1.0)  # good customers lost to it
    q_analyst: float = Field(default=0.91, ge=0.0, le=1.0)  # analyst agreement rate
    c_review: float = Field(default=6.77, ge=0.0)  # $58/hr loaded, 7 min handling
    d_delay: float = Field(default=1.20, ge=0.0)  # delay cost per reviewed case

    # --- operational constraints -------------------------------------------------
    daily_review_slots: int = Field(default=60, ge=0)  # ~1 analyst FTE
    cost_per_decision: float = Field(default=0.0008, ge=0.0)  # infra, per scored event

    # --- relationship cost of declining a good customer --------------------------
    # BUSINESS ASSUMPTION, and the softest number in the system: P(churn | declined)
    # has no data support in IEEE-CIS. It is graded by tenure from published
    # false-decline research. It drives the whole false-positive cost term, therefore
    # every decision boundary. Owner review required whenever the boundaries are
    # re-derived (design spec §11).
    p_churn_on_decline: dict[str, float] = Field(
        default_factory=lambda: {
            "new(0d)": 0.42,
            "1-7d": 0.38,
            "8-30d": 0.30,
            "31-90d": 0.22,
            "91-180d": 0.16,
            "181-400d": 0.12,
            "400d+": 0.09,
        }
    )

    # MEASURED, not assumed: residual lifetime value per tenure bucket, out of
    # `research/04b_ltv.py` (P(repeat) x annualised GMV if repeat x margin x 0.85
    # survival discount), persisted as `data/tenure_econ.csv`. Pinned here at full
    # precision so the economics module needs no I/O; refresh it when the LTV study is
    # re-run, and expect every break-even boundary to move when you do.
    residual_ltv: dict[str, float] = Field(
        default_factory=lambda: {
            "new(0d)": 323.4342016540025,
            "1-7d": 540.7529669064562,
            "8-30d": 425.01190539855946,
            "31-90d": 364.1090248849848,
            "91-180d": 347.5898044409378,
            "181-400d": 300.73712726342706,
            "400d+": 367.39809974090804,
        }
    )

    @model_validator(mode="after")
    def _check_coherent(self) -> BusinessConstants:
        # Margin is defined as 1 - COGS. Overriding one without the other silently
        # prices fraud losses and false declines off different revenue models, which
        # is invisible in the output and wrong in every number downstream.
        if abs(self.cogs + self.margin - 1.0) > 1e-12:
            raise ValueError(f"cogs + margin must equal 1.0, got {self.cogs + self.margin}")
        missing = set(TENURE_LABELS) - set(self.p_churn_on_decline) - set()
        if missing or set(TENURE_LABELS) - set(self.residual_ltv):
            raise ValueError("p_churn_on_decline and residual_ltv must cover every tenure bucket")
        return self


# Read once at import, 12-factor style: the process is configured at startup and the
# configuration cannot then drift underneath a running decision. Functions take this as
# a default argument so they stay pure and a test can pass a different set explicitly.
SETTINGS = BusinessConstants()
