import pandas as pd, numpy as np
from config import (F_PASS, A_ABANDON, Q_ANALYST, C_REVIEW, D_DELAY,
                    DAILY_REVIEW_SLOTS, COST_PER_DECISION, SENS_F, SENS_CD,
                    SEED_LABEL_SIM, CHARGEBACK_MEDIAN_DAYS, CHARGEBACK_LOG_SIGMA, TRAIN_END)
pd.set_option('display.width', 220)
d = pd.read_parquet('data/econ_test.parquet')
L, M, y = d.L.values, d.M.values, d.isFraud.values
p, p_bal, p_raw = d.p.values, d.p_bal.values, d.p_raw.values
DAYS = d.day.nunique(); ANN = 365/DAYS
F, Aa, Q, C, D = F_PASS, A_ABANDON, Q_ANALYST, C_REVIEW, D_DELAY

def evs(p,L,M): return np.vstack([-p*L, -(p*F*L+(1-p)*Aa*M),
    -((1-Q)*(p*L+(1-p)*M)+C+D), -(1-p)*M])
def realised(a,y,L,M):
    c=np.zeros(len(y))
    m=a==0; c[m]=y[m]*L[m]
    m=a==1; c[m]=y[m]*F*L[m]+(1-y[m])*Aa*M[m]
    m=a==2; c[m]=(1-Q)*(y[m]*L[m]+(1-y[m])*M[m])+C+D
    m=a==3; c[m]=(1-y[m])*M[m]
    return c

print("="*78); print("1. WHAT MISCALIBRATION COSTS, ON REAL DATA")
print("="*78)
rows=[]
for nm,ps in [('calibrated (isotonic)',p),('raw uncalibrated',p_raw),
              ('class-rebalanced, uncorrected',p_bal)]:
    a = evs(ps,L,M).argmax(0)
    c = realised(a,y,L,M).sum()*ANN
    mix = pd.Series(a).value_counts(normalize=True).reindex([0,1,2,3]).fillna(0)
    rows.append(dict(score=nm, annual_cost=c, allow=mix[0], challenge=mix[1],
                     review=mix[2], deny=mix[3],
                     fraud_allowed=((a==0)&(y==1)).sum(), good_denied=((a==3)&(y==0)).sum()))
R=pd.DataFrame(rows)
R['penalty_vs_calibrated']=R.annual_cost-R.annual_cost[0]
sh=R.copy()
for c_ in ['annual_cost','penalty_vs_calibrated']: sh[c_]=sh[c_].map(lambda v:f"${v:,.0f}")
for c_ in ['allow','challenge','review','deny']: sh[c_]=sh[c_].map(lambda v:f"{v:.2%}")
print(sh.to_string(index=False))
print("\nAll three scores have essentially the same AUC (0.9045-0.9050).")

print("\n"+"="*78); print("2. CAPACITY-CONSTRAINED REVIEW: rank by value-of-review, not by score")
print("="*78)
e = evs(p,L,M)
auto = e[[0,1,3]].max(0)                     # best action without an analyst
vor  = e[2]-auto                             # value of sending it to a human
daily_slots = DAILY_REVIEW_SLOTS                             # ~1 analyst FTE at 64 cases/day
budget = daily_slots*DAYS
print(f"analyst budget: {daily_slots}/day x {DAYS}d = {budget:,} cases "
      f"({budget/len(d):.2%} of volume)")
for nm, rank in [('rank by score p (typical case mgmt default)', p),
                 ('rank by uncertainty p(1-p)', p*(1-p)),
                 ('rank by uncertainty x exposure', p*(1-p)*d.TransactionAmt.values),
                 ('rank by value-of-review (correct)', vor)]:
    sel = np.zeros(len(d),bool); sel[np.argsort(-rank)[:budget]]=True
    a = e[[0,1,3]].argmax(0); a = np.where(a==2,3,a)   # map back to 0/1/3
    a = np.where(sel, 2, a)
    c = realised(a,y,L,M).sum()*ANN
    fr = y[sel].mean()
    print(f"  {nm:<46} ${c:,.0f}/yr   fraud in queue {fr:.1%}   "
          f"realised VoR ${vor[sel].sum():,.0f}")
print(f"\nvalue-of-review is positive for only {(vor>0).sum():,} of {len(d):,} transactions "
      f"({(vor>0).mean():.3%}) at c+d=${C+D:.2f}")

print("\n"+"="*78); print("3. SENSITIVITY: when does human review earn its place?")
print("="*78)
grid=[]
for f_ in SENS_F:
    for cd in SENS_CD:
        ee = np.vstack([-p*L, -(p*f_*L+(1-p)*Aa*M),
                        -((1-Q)*(p*L+(1-p)*M)+cd), -(1-p)*M])
        a = ee.argmax(0)
        grid.append(dict(f=f_, cost_review=cd, pct_review=(a==2).mean(),
                         annual=realised(a,y,L,M).sum()*ANN))
G=pd.DataFrame(grid)
print(G.pivot(index='f',columns='cost_review',values='pct_review').map(lambda v:f"{v:.2%}"))
print("\n(share of volume routed to human review)\n")
print(G.pivot(index='f',columns='cost_review',values='annual').map(lambda v:f"${v/1e6:.2f}M"))
print("(annualised total cost)")

print("\n"+"="*78); print("4. THE LABEL-LATENCY BUG, SIMULATED")
print("="*78)
np.random.seed(SEED_LABEL_SIM)
X = pd.read_parquet('data/X.parquet', columns=['isFraud','split','day'])
tr = X[X.split=='train'].copy()
lag = np.random.lognormal(np.log(CHARGEBACK_MEDIAN_DAYS), CHARGEBACK_LOG_SIGMA, len(tr))       # median 34d chargeback arrival
tr['lag']=lag; tr['arrived'] = (tr.day + tr.lag) <= TRAIN_END-1     # what we'd know on day 119
naive = np.where(tr.isFraud==1, tr.arrived.astype(int), 0)  # unarrived -> coded 0
print(f"true fraud in train      : {tr.isFraud.sum():,} ({tr.isFraud.mean():.3%})")
print(f"labels actually arrived  : {naive.sum():,} ({naive.mean():.3%})  "
      f"-> {1-naive.sum()/tr.isFraud.sum():.1%} of fraud invisible at training time")
tr['naive']=naive; tr['block']=tr.day//20
cmp = tr.groupby('block').agg(true_rate=('isFraud','mean'), observed_rate=('naive','mean'))
cmp['ratio']=cmp.observed_rate/cmp.true_rate
print("\nfraud rate the naive model would see, by 20-day block of the training window:")
print(cmp.round(4).to_string())
print("\nThe apparent fraud rate falls off a cliff in recent data purely because the")
print("chargebacks have not arrived yet. A model trained on this learns 'recent = safe'.")

print("\n"+"="*78); print("5. THE FIVE-TERM P&L")
print("="*78)
a = evs(p,L,M).argmax(0)
fraud_loss = (y*(a==0)*L + y*(a==1)*F*L + (1-Q)*y*(a==2)*L).sum()*ANN
friction   = ((1-y)*(a==3)*M + (1-y)*(a==1)*Aa*M + (1-Q)*(1-y)*(a==2)*M).sum()*ANN
ops        = ((a==2)*(C+D)).sum()*ANN
infra      = len(d)*COST_PER_DECISION*ANN
tot = fraud_loss+friction+ops+infra
for nm,v in [('Fraud losses',fraud_loss),('Friction (declines + abandons)',friction),
             ('Operations (analyst)',ops),('Infrastructure',infra),('TOTAL',tot)]:
    print(f"  {nm:<34} ${v:>12,.0f}   {v/tot:>6.1%}")
print(f"\nfriction / fraud ratio = {friction/fraud_loss:.2f}x")
