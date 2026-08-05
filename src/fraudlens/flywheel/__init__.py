"""The retraining loop: when to retrain, how to try a challenger, when to promote, how to undo.

The whole package is organised around one asymmetry. **Retraining is triggered on Tier 0
signals that are available today; promotion is decided on Tier 2 signals that are available
in a quarter.** Those are different questions at different latencies and the standard
mistake is to answer both with the same number.

Concretely: "AUC dropped, retrain" is unimplementable here. On a 30-day-old window the
observed fraud rate is 6.9% of the true rate (findings §6), so the AUC that triggered the
retrain was measured on a population whose labels are mostly still in the post. Worse, the
error is directional — the unrevealed rows read as clean, so the more permissive model
always looks cheapest. A trigger built on that fires late, fires on noise, and when it does
fire it hands the retrainer a training window with the same bias baked into it.

So `trigger` reads drift, mix and staleness — none of which need a label — and `promotion`
reads matured cost, which needs a quarter. `shadow` is what fills the gap between them: a
challenger that has been scoring live traffic since the trigger fired has a paired cost
comparison ready the moment the labels mature, instead of starting the 90-day clock at the
point somebody asks for a promotion.

`rollback` exists because the ledger is append-only and a promoted model has already
decided real money. Undo is not available; identifying and pricing the affected population
is.

Deliberately empty of re-exports, matching `models` and `monitoring`: a forwarding
`__init__` hides which module a symbol lives in, which is the first thing a reader needs.
"""
