-- 001: the decision ledger.
--
-- ADR-0002 ranks auditability second, above latency. Every column here exists
-- so that a single declined checkout can be reconstructed after the customer
-- disputes it: what the model said, which model said it, which policy turned
-- that score into an action, and what the inputs were.
--
-- transaction_id is the primary key rather than a surrogate. That is the
-- idempotency mechanism: an at-least-once replay collides instead of
-- double-writing, and no application-side dedupe is trusted.

CREATE TABLE IF NOT EXISTS decision_ledger (
    transaction_id          BIGINT       PRIMARY KEY,
    transaction_at          TIMESTAMPTZ  NOT NULL,
    decided_at              TIMESTAMPTZ  NOT NULL,
    -- Nullable on purpose. A degraded decision is made by the rule ladder
    -- without a model, so there is no score and no probability. Writing 0.0
    -- would be a fabricated observation indistinguishable from a genuine
    -- near-zero one, and it would drag every drift, calibration and
    -- score-distribution metric toward zero in proportion to the outage —
    -- the monitoring would look calmest exactly when the model was most
    -- broken. NULL is the honest encoding; the CHECK below makes it
    -- structural rather than conventional.
    score                   DOUBLE PRECISION,
    calibrated_probability  DOUBLE PRECISION,
    action                  TEXT         NOT NULL,
    reason_codes            JSON         NOT NULL,
    model_version           TEXT         NOT NULL,
    policy_version          TEXT         NOT NULL,
    feature_version         TEXT         NOT NULL,
    config_hash             TEXT         NOT NULL,
    input_hash              TEXT         NOT NULL,
    degraded                BOOLEAN      NOT NULL,
    degraded_reason         TEXT,

    CONSTRAINT ck_decision_probability_range
        CHECK (calibrated_probability IS NULL
            OR (calibrated_probability >= 0 AND calibrated_probability <= 1)),
    CONSTRAINT ck_decision_action
        CHECK (action IN ('allow', 'challenge', 'review', 'deny')),
    -- A scored decision must carry its score, and a degraded one must not
    -- invent one. Without this, "probability IS NULL" degrades from a
    -- guarantee to a convention, and the filter that protects every
    -- downstream metric stops being enforceable.
    CONSTRAINT ck_decision_score_presence
        CHECK ((degraded = false AND calibrated_probability IS NOT NULL AND score IS NOT NULL)
            OR (degraded = true  AND calibrated_probability IS NULL     AND score IS NULL)),
    -- A degraded decision has to be excludable from downstream analysis on
    -- evidence, not on recollection. Flagged-but-unexplained is rejected.
    CONSTRAINT ck_decision_degraded_reason
        CHECK ((degraded = false AND degraded_reason IS NULL)
            OR (degraded = true  AND degraded_reason IS NOT NULL))
)
--;;
CREATE INDEX IF NOT EXISTS ix_decision_ledger_transaction_at
    ON decision_ledger (transaction_at)
