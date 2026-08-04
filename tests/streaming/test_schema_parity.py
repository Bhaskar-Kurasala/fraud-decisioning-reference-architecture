"""The SQL migrations and the Core metadata must describe the same tables.

Postgres is built from the .sql files; SQLite is built from the metadata. Two
descriptions of one schema is a drift risk, and this test is the price of
keeping both — without it, a column added for the audit trail could exist in
tests and be missing in production, which is the worst possible direction.
"""

from __future__ import annotations

import re

from fraudlens.streaming.migrate import migration_files
from fraudlens.streaming.schema import metadata

_CREATE_TABLE = re.compile(
    r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\)", re.DOTALL | re.IGNORECASE
)


# Words that begin a line inside a CREATE TABLE body without naming a column:
# constraint clauses and their wrapped continuations.
_NOT_COLUMNS = {
    "CONSTRAINT",
    "PRIMARY",
    "FOREIGN",
    "UNIQUE",
    "CHECK",
    "REFERENCES",
    "AND",
    "OR",
}


def columns_declared_in_sql(body: str) -> set[str]:
    names: set[str] = set()
    for raw in body.split("\n"):
        line = raw.split("--")[0].strip()
        match = re.match(r"^(\w+)\s+\S", line)
        if match and match.group(1).upper() not in _NOT_COLUMNS:
            names.add(match.group(1))
    return names


def test_sql_migrations_declare_the_same_columns_as_the_core_metadata() -> None:
    found: dict[str, set[str]] = {}
    for path in migration_files():
        for table, body in _CREATE_TABLE.findall(path.read_text(encoding="utf-8")):
            found[table] = columns_declared_in_sql(body)

    assert set(found) == set(metadata.tables), "a table exists in one place only"
    for name, sql_columns in found.items():
        assert sql_columns == {c.name for c in metadata.tables[name].columns}, name


def test_every_migration_is_numbered_and_ordered() -> None:
    names = [p.name for p in migration_files()]
    assert names == sorted(names)
    assert all(re.match(r"^\d{3}_\w+\.sql$", n) for n in names), names
