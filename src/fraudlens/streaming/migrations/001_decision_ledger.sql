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
    score                   DOUBLE PRECISION NOT NULL,
    calibrated_probability  DOUBLE PRECISION NOT NULL,
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
        CHECK (calibrated_probability >= 0 AND calibrated_probability <= 1),
    CONSTRAINT ck_decision_action
        CHECK (action IN ('allow', 'challenge', 'review', 'deny')),
    -- A degraded decision has to be excludable from downstream analysis on
    -- evidence, not on recollection. Flagged-but-unexplained is rejected.
    CONSTRAINT ck_decision_degraded_reason
        CHECK ((degraded = false AND degraded_reason IS NULL)
            OR (degraded = true  AND degraded_reason IS NOT NULL))
)
--;;
CREATE INDEX IF NOT EXISTS ix_decision_ledger_transaction_at
    ON decision_ledger (transaction_at)
