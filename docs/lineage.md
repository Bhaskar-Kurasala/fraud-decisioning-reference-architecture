# Lineage: what we can prove, and what we cannot

ADR-0002 ranks auditability second, above latency. That ranking only means something if
someone has worked out what the audit trail actually supports. This document is that
working-out: which question each recorded stamp answers, where the chain of custody
breaks today, and what closing each break would cost.

The short version: **the policy arm of a decision is reproducible and asserted by test;
the model arm is not reproducible and is not claimed to be.** Everything below is detail
on that sentence.

---

## 1. What is stamped, and what question it answers

Every row in `decision_ledger` carries five identifiers beyond the decision itself.

| Stamp | The question it answers | How well it answers it |
|---|---|---|
| `model_version` | "Which model declined me?" | Names the artifact. Does **not** pin its contents — see gap 1. |
| `policy_version` | "Which decision rule ran?" | Selects the policy variant. Free-form string, no registry — gap 4. |
| `feature_version` | "What contract were the inputs supplied under?" | `request-supplied-v1`: there is no feature store, the caller assembles the vector. It versions the *contract*, not a pipeline. |
| `config_hash` | "Was the business config different that day?" | Yes/no, reliably. **Not** *what* was different — gap 3. |
| `input_hash` | "Are we looking at the same transaction we decided on?" | Detects a substituted input. Does not let us recover the input — deliberate, see §3. |

Plus `degraded` / `degraded_reason`, which answer "did a model decide this at all, or did
the fail-safe ladder?" — the question that has to be asked before any of the others are
worth asking.

Around the ledger sit two more records:

- **The MLflow run** (`fraudlens.models.tracking`) holds the metrics, the params and the
  provenance triple (git SHA, config hash, data checksum) of the training run, plus the
  model signature.
- **The reproducibility manifest** (`fraudlens.lineage.manifest`) pins what it took to
  produce one published artifact: git SHA *and dirty-tree state*, config hash, data
  checksum, seeds, the `uv.lock` digest, the Python version, and the single command that
  regenerates it.

The dirty-tree flag is the field that earns the manifest its place. An artifact built
from a modified working tree is not reproducible, and recording the last commit anyway —
which is what a bare `git rev-parse HEAD` does — produces a record pointing confidently
at code that was never executed. The manifest reports that state as a caveat and
`reproducible` is False. A negative that is legible beats a silence.

---

## 2. The replay claim, stated precisely

§9a: *"A decision in the ledger can be replayed: same inputs + same recorded model and
policy versions must produce the same output. This is asserted by a test, not assumed."*

`fraudlens.lineage.replay.replay_decision` is the executable form.
`tests/lineage/test_replay.py` is the assertion — a decision produced by the real serving
path, written to a real ledger, and re-derived from its own row, on every arm of the
three-action policy.

What it re-runs: the recorded calibrated probability, plus the original request, back
through `economics` and `policy.decide` under the recorded `policy_version`. A change to
a cost function, a business constant or the policy variant shows up here as a diff in
the action.

What it does **not** re-run: features → raw score → calibrated probability. The result
distinguishes four outcomes, and the distinction is the product:

- `reproduced` — same input, same config, same action.
- `unverified` — the supplied input does not hash to `input_hash`. With three actions,
  two unrelated transactions agree by chance far too often for a bare action match to be
  evidence; this is why `ReplayResult.verified` requires the input to match and
  `matched` alone is not enough.
- config drift — the action differs and `config_hash` differs. A change-management
  finding, not a defect.
- `DIVERGENCE` — the action differs on the same input under the same config. The
  decision path no longer agrees with its own audit trail. This is the only outcome that
  should page anyone.

---

## 3. Why the features are not in the ledger

`input_hash` is a digest of the request, not a copy of it. That is a deliberate
trade and it is the source of the largest gap below, so the reasoning should be on the
record:

- Copying the feature vector into an append-only table creates a second permanent store
  of customer-linked signals **with no deletion path**, because the table is append-only
  by design and by trigger. ADR-0002 already declines to build the GDPR erasure workflow;
  building a store that makes erasure structurally impossible is a different and worse
  decision.
- The features are the caller's data. The caller assembles them (`FEATURE_VERSION =
  request-supplied-v1`) and already has them.

The cost is exact: replay requires the caller to supply the input, and the ledger can
only *check* that they supplied the right one. Where the caller no longer holds the
request, the decision is not replayable at all. That is the price of not building an
undeletable copy, and it is the right price — but it is a price, not a free choice.

---

## 4. The gap list

Ordered by how much a dispute or an audit would care.

### Gap 1 — `model_version` names an artifact but does not pin its contents

The string comes from the loaded scorer. Nothing binds it to a content digest, so two
deployments can serve different weights under the same version string and the ledger
cannot tell them apart. "Which model declined me" is answerable to the precision of a
label someone chose, not of an artifact.

**Cost to close:** add an artifact digest to the ledger row — the MLflow model version's
`source` plus a hash of the serialised model, computed once at load and carried on the
scorer. One column, one migration, one field on the `CalibratedScorer` protocol. This is
the cheapest high-value item on this list and should be done first.

### Gap 2 — degraded decisions cannot be replayed at all — **CLOSED**

*Closed after this document was first written. Kept here rather than deleted, because the
reason it existed is more instructive than the fix.*

The fail-safe rule ladder lived in `serving.decisioning`, which sits *above* `lineage` in
the import contract, so `replay` could not call it — and reimplementing it from its prose
is exactly the mistake that has already cost this project two corrections ($71,545/yr and
$583/yr). So degraded rows were reported unreplayable.

The consequence was worst precisely where it mattered most. An outage is when the largest
block of unusual decisions gets made, it is the block a regulator asks about first, and it
was the one block nobody could later prove was decided correctly. Note that this was not
an oversight in the ladder — it was a consequence of where the *layers* were drawn, which
is the kind of defect that is invisible until something downstream tries to use the layer
below.

**Closed by** moving the ladder into `policy.fallback` with its own version string
(`rules-ladder-v1`), which both `serving` and `lineage` now call. `POLICY_VERSION` names
both policies the deployment can decide under — `ev-argmax-3action-v1+rules-ladder-v1` —
because the ledger has one column to say which arithmetic applied and a scored row and a
degraded row did not come from the same one.

Two things fell out of the move that were not the point of it:

- `serving.reasons` had its own `_HIGH_AMOUNT = 500.0` beside a comment claiming the
  amount bands "carry no money — moving them changes what a dispute letter says, not what
  we decide". That was false: the same threshold is the deny rung of the ladder. The band
  now takes its value from `policy.fallback`, so the dispute letter's notion of "high
  amount" is necessarily the one the ladder acted on.
- The `FALLBACK_RULE_*` reason codes now take their string values from the policy module
  that owns the rungs, rather than being a second set of literals that agree until someone
  edits one.

Replay now covers 100% of ledger rows.

### Gap 3 — `config_hash` proves *different*, never *what*

The ledger stores the hash of the business constants, not the constants. So "was the
config different on 3 June" is answerable and "what was different" is not. In a dispute
that is the difference between "our records show a change" and "our records show
`P_CHURN_ON_DECLINE` for new accounts moved from 0.42 to 0.30, which moved your
break-even from 0.64 to 0.71".

**Cost to close:** a `config_versions` table — `config_hash` primary key, canonical JSON
body, `first_seen_at` — written on service start with an upsert. Perhaps 40 lines. It
also gives the manifest something to resolve a historical hash against, which nothing
can do today.

### Gap 4 — `policy_version` and `feature_version` are unregistered free strings

Nothing structurally prevents two builds from writing the same `policy_version` while
behaving differently. `lineage.replay.POLICY_VARIANTS` is a partial fix: it maps the
version to the policy variant and refuses an unknown one rather than replaying it under
today's policy, and `test_serving_policy_version_is_registered` fails if serving's
constants drift away from the table. That closes the *drift between the two modules*. It
does not close *drift between two deployed builds*.

**Cost to close:** derive the version string, or a component of it, from a hash of the
policy definition, so the string cannot stay constant while the behaviour changes. Needs
care — a hash that changes on every whitespace edit makes every historical row
unreplayable, which is worse.

### Gap 5 — the ledger is append-only by trigger, not sealed

`migrations/003` and `schema.sqlite_append_only_ddl` block UPDATE and DELETE. An operator
with DDL privileges can drop the trigger, edit a row, and recreate it. So "can we prove
the policy was not changed retroactively" is answerable **only under an assumption of
database access control**, which is an assumption about people, not a property of the
system.

**Cost to close:** hash-chain the rows (each row carries the digest of the previous one)
plus a periodic digest anchored somewhere the DB operator does not control. One column, a
verifier, and an operational routine for the anchor. Worth doing before this system is
ever used as evidence in an actual dispute; not worth doing before that.

### Gap 6 — the research pipeline emits no manifest

`bash run_all.sh` regenerates every number in `fraud-decisioning-findings.md`, which
satisfies §9a's "single documented command". But it writes no manifest, so the published
figures have no machine-readable provenance record — only the model card has one. Nothing
records which commit or which raw-CSV checksum a given published table came from.

**Cost to close:** call `build_manifest` at the end of each research stage and write the
JSON beside the output. The scripts are preserved unmodified under ADR-0001, so this
would be a wrapper stage rather than an edit to them.

### Gap 7 — one published number cannot be regenerated at all

The naive $1,999 LTV figure was produced by an `04_ltv.py` that no longer exists
(verification report §3.2). `run_all.sh` cannot regenerate it; the closest reconstruction
is $1,984.16. §9a's guarantee therefore has a known hole that predates this epic, and the
model card states the effect as "roughly 6x" rather than quoting the number — which is
the honest treatment of a figure with no generating code.

**Cost to close:** it cannot be closed retroactively. The number should be quoted as an
order of magnitude or dropped.

### Gap 8 — the model card's run is not shareable

`mlruns/` and `data/` are git-ignored. `scripts/regenerate_model_card.py` regenerates the
run and the card deterministically from `data/scored_test.parquet`, so the card *is*
reproducible — but the run id it names points into a local store nobody else has, and two
people regenerating the card get different run ids for the same numbers.

**Cost to close:** a shared MLflow backend (the compose stack in E10 has one), or export
the run alongside the card. Until then the card's numbers are checkable and its run
pointer is not.

### Gap 9 — the card's limitations are code, not a live reference

`model_card._STANDING_LIMITATIONS` is a tuple of strings maintained beside the renderer.
If `docs/findings/verification-report.md` is revised, the card keeps the old wording and
nothing fails. It is quoted at the findings' own confidence today ("roughly 6x", not a
measured figure); staying that way is currently a matter of discipline.

**Cost to close:** a golden test linking each limitation to a stable anchor in the
findings document. Cheap, but it needs the findings doc to carry anchors, which is
outside this epic's scope.

### Gap 10 — human adjudications had no label provenance — **CLOSED**

*Closed by E12c. Kept here rather than deleted, because the failure mode it
prevents is not obvious.*

`revealed_labels` held one label per transaction, keyed on `transaction_id`,
arriving via chargeback at a 34-day median. But humans also adjudicate
transactions — dispute handlers, representment, and the small audit queue E12b
priced as a control-chart instrument. Those adjudications are labels too, and
they arrive ~90 days before the chargeback they may or may not prevent.

Without provenance, the two would mix silently. A human "fraud" call is an
*opinion* correct at q=0.91 and drawn from a censored sample above the decision
boundary; a chargeback is an *outcome*. Mixing them in training changes the
sampling distribution of the training set; mixing them in the promotion gate
compares challenger cost on rows whose labels carry 9% error against rows whose
labels carry none. Both are silent corruptions, and both are directional — the
human label biases toward the opinion, which in the censored region is always
more suspicious than the population.

**Closed by** a separate `human_adjudications` table (same reasoning as
`shadow_scores`: mixing two kinds of observation in one table means every
consumer has to remember to filter, and the one that forgets corrupts
silently). The default training/promotion path reads `revealed_labels` and
structurally cannot see human labels. Including them is an explicit opt-in
through `effective_labels(..., include_human=True)`, which tags every label
with its origin so the consumer can still tell them apart.

The reconciliation rule: **chargeback wins when both exist.** The outcome is
ground truth; the opinion was an early guess. The disagreement is recorded —
it is the control-chart signal. The prevented-loss case (human says "fraud",
no chargeback arrives because the decline worked) is left as `origin=HUMAN`,
excluded from training, because it cannot be distinguished from a false
positive and assuming it is always a prevented loss would inflate the fraud
rate by the analyst false-positive rate.

---

## 5. What an auditor can and cannot be told

**Can be told, with evidence:**

- Every decision this service made, once, in an append-only table, with its action, its
  reason codes and the calibrated probability it was derived from.
- Whether a model or the fail-safe ladder produced it, and why the ladder ran — and for
  ladder decisions, that the recorded action falls out of the recorded request through the
  same ladder the service ran, on every rung (gap 2, closed).
- That the recorded action falls out of the recorded probability, the original request
  and the recorded config — mechanically, by test, on every arm of the policy.
- Whether the business config in force at decision time differs from today's.
- For the champion model: its measured performance, the windows it was trained,
  calibrated and evaluated on, and the commit, config and data checksum behind the card —
  including when one of those pins was missing.
- That every label feeding training or the promotion gate is a chargeback outcome, not a
  human opinion, and that the two label sources are separated by table not by convention
  (gap 10, closed).

**Cannot be told, today:**

- That a specific set of weights produced a specific score (gap 1).
- What the config was, as opposed to whether it changed (gap 3).
- That no privileged operator altered the ledger (gap 5).
- Where a published research figure came from, at the level of a commit and a data
  checksum (gap 6), and in one case at all (gap 7).

An auditor who is told the first list and shown the second is in a materially better
position than one shown a diagram claiming end-to-end lineage. The gaps are the
deliverable.
