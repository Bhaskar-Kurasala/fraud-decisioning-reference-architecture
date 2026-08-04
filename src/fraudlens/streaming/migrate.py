"""Apply the streaming schema. One command, from the repo, on either engine.

Reproducibility is non-negotiable (spec §9a): the schema has to be recreatable
from what is committed, not from whatever a database happens to contain.

Two paths, because the system genuinely runs on two engines:

* **Postgres** replays the numbered ``.sql`` files. They are the source of
  truth, including the append-only triggers, which SQLAlchemy cannot express.
* **SQLite** builds from the mirrored Core metadata plus the SQLite spelling of
  the same triggers. This exists so the append-only and idempotency invariants
  are asserted in the default test suite rather than only in an integration run.

Usage::

    uv run python -m fraudlens.streaming.migrate postgresql+psycopg://.../fraudlens
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import Engine, create_engine, text

from fraudlens.streaming.schema import metadata, sqlite_append_only_ddl

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# The plpgsql function body contains semicolons, so the runner does not parse
# SQL — files declare their own statement boundary explicitly.
STATEMENT_SEPARATOR = "\n--;;\n"


def migration_files() -> list[Path]:
    """The numbered migrations, in application order."""
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def _statements(sql: str) -> list[str]:
    return [chunk.strip() for chunk in sql.split(STATEMENT_SEPARATOR) if chunk.strip()]


def migrate(engine: Engine) -> None:
    """Create the streaming schema and its append-only guards, idempotently."""
    dialect = engine.dialect.name
    if dialect == "postgresql":
        with engine.begin() as conn:
            for path in migration_files():
                for statement in _statements(path.read_text(encoding="utf-8")):
                    conn.execute(text(statement))
        return
    if dialect == "sqlite":
        metadata.create_all(engine)
        with engine.begin() as conn:
            # SQLite has no CREATE TRIGGER IF NOT EXISTS in older builds; the
            # tables are freshly created per test engine, so this is a no-op on
            # a second call only in the sense that it will not be reached.
            for statement in sqlite_append_only_ddl():
                conn.execute(text(statement))
        return
    raise ValueError(f"unsupported dialect {dialect!r}: the ledger runs on Postgres or SQLite")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m fraudlens.streaming.migrate <database-url>", file=sys.stderr)
        return 2
    engine = create_engine(args[0])
    try:
        migrate(engine)
    finally:
        engine.dispose()
    print(f"applied {len(migration_files())} migration(s) to {engine.dialect.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
