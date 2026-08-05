-- 004: human adjudications — a label source that is an opinion, not an outcome.
--
-- A chargeback is ground truth: a bank reversed the charge because the
-- customer proved fraud. A human adjudication is an analyst's call, correct
-- at the measured q_analyst = 0.91 and drawn from a censored, non-random
-- sample (the queue sits above the decision boundary; E12b showed its lowest
-- case at p=0.375 against a median boundary of 0.089).
--
-- Mixing the two in revealed_labels would corrupt both training and the
-- promotion gate: the gate would compare challenger cost on rows whose labels
-- carry 9% error against rows whose labels carry none, and the training set
-- would absorb a sampling distribution that has nothing to do with the
-- population. Separate table; the default query from revealed_labels cannot
-- see these rows.

CREATE TABLE IF NOT EXISTS human_adjudications (
    transaction_id     BIGINT       NOT NULL
                                    REFERENCES decision_ledger (transaction_id),
    is_fraud           BOOLEAN      NOT NULL,
    adjudicated_at     TIMESTAMPTZ  NOT NULL,
    adjudicator        TEXT         NOT NULL,

    PRIMARY KEY (transaction_id, adjudicator)
)
--;;
CREATE INDEX IF NOT EXISTS ix_human_adjudications_adjudicated_at
    ON human_adjudications (adjudicated_at)
--;;
DROP TRIGGER IF EXISTS human_adjudications_append_only ON human_adjudications
--;;
CREATE TRIGGER human_adjudications_append_only
    BEFORE UPDATE OR DELETE ON human_adjudications
    FOR EACH STATEMENT EXECUTE FUNCTION fraudlens_deny_mutation()
--;;
DROP TRIGGER IF EXISTS human_adjudications_no_truncate ON human_adjudications
--;;
CREATE TRIGGER human_adjudications_no_truncate
    BEFORE TRUNCATE ON human_adjudications
    FOR EACH STATEMENT EXECUTE FUNCTION fraudlens_deny_mutation()
--;;
REVOKE UPDATE, DELETE, TRUNCATE ON human_adjudications FROM PUBLIC
