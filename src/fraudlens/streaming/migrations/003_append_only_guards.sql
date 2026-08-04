-- 003: append-only enforcement, at the database.
--
-- ADR-0002 puts constraints in the database. Application-level discipline is
-- not an audit control: it is bypassed by a psql session, a migration script,
-- or the next engineer. A ledger that *can* be edited is evidence of nothing,
-- so the edit has to be impossible for the role the application connects as.
--
-- Two layers, deliberately:
--   1. REVOKE removes the privilege from the application role.
--   2. The trigger fires even for the table owner, whom REVOKE does not bind.
--      Restoring mutability then requires dropping a trigger, which is itself
--      a visible, reviewable DDL change.
--
-- Statements are separated by `--;;` because the plpgsql body contains
-- semicolons and the runner does not parse SQL.

CREATE OR REPLACE FUNCTION fraudlens_deny_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION '% is append-only: % denied', TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql
--;;
DROP TRIGGER IF EXISTS decision_ledger_append_only ON decision_ledger
--;;
-- Postgres does not allow TRUNCATE in the same trigger as UPDATE/DELETE, hence
-- the pair per table.
CREATE TRIGGER decision_ledger_append_only
    BEFORE UPDATE OR DELETE ON decision_ledger
    FOR EACH STATEMENT EXECUTE FUNCTION fraudlens_deny_mutation()
--;;
DROP TRIGGER IF EXISTS decision_ledger_no_truncate ON decision_ledger
--;;
CREATE TRIGGER decision_ledger_no_truncate
    BEFORE TRUNCATE ON decision_ledger
    FOR EACH STATEMENT EXECUTE FUNCTION fraudlens_deny_mutation()
--;;
DROP TRIGGER IF EXISTS revealed_labels_append_only ON revealed_labels
--;;
CREATE TRIGGER revealed_labels_append_only
    BEFORE UPDATE OR DELETE ON revealed_labels
    FOR EACH STATEMENT EXECUTE FUNCTION fraudlens_deny_mutation()
--;;
DROP TRIGGER IF EXISTS revealed_labels_no_truncate ON revealed_labels
--;;
CREATE TRIGGER revealed_labels_no_truncate
    BEFORE TRUNCATE ON revealed_labels
    FOR EACH STATEMENT EXECUTE FUNCTION fraudlens_deny_mutation()
--;;
REVOKE UPDATE, DELETE, TRUNCATE ON decision_ledger FROM PUBLIC
--;;
REVOKE UPDATE, DELETE, TRUNCATE ON revealed_labels FROM PUBLIC
