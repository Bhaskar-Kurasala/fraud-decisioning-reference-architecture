-- 002: delayed labels.
--
-- Separate table, joined on transaction_id, because a label is a different
-- event from a decision and arrives 30-90 days later. Keeping it out of
-- decision_ledger is what lets that table stay append-only: recording the
-- outcome never mutates the decision.
--
-- A row exists here only once the label has actually been revealed. That is
-- the leak guard: there is no "pending" state in the database that a careless
-- join could read early.

CREATE TABLE IF NOT EXISTS revealed_labels (
    transaction_id     BIGINT       PRIMARY KEY
                                    REFERENCES decision_ledger (transaction_id),
    is_fraud           BOOLEAN      NOT NULL,
    transaction_at     TIMESTAMPTZ  NOT NULL,
    -- Simulated time the dispute would have landed.
    revealed_at        TIMESTAMPTZ  NOT NULL,
    -- Wall-clock time this process wrote the row. Reconciling the two is how
    -- an auditor proves the revealer was not run ahead of its own schedule.
    recorded_at        TIMESTAMPTZ  NOT NULL,
    dispute_lag_days   DOUBLE PRECISION NOT NULL,
    label_source       TEXT         NOT NULL,

    CONSTRAINT ck_label_lag_non_negative CHECK (dispute_lag_days >= 0),
    CONSTRAINT ck_label_reveal_after_transaction CHECK (revealed_at >= transaction_at)
)
--;;
CREATE INDEX IF NOT EXISTS ix_revealed_labels_revealed_at
    ON revealed_labels (revealed_at)
