import pandas as pd, numpy as np, json
from config import (COGS, MARGIN, CB_FEE, OPS_DISPUTE, F_PASS, A_ABANDON,
                    Q_ANALYST, C_REVIEW, D_DELAY, TENURE_EDGES, TENURE_LABELS, DATA)
OPS_DISP = OPS_DISPUTE
pd.set_option('display.width', 220)

d = pd.read_parquet('data/scored_test.parquet')
ten = pd.read_csv('data/tenure_econ.csv', index_col=0)

DAYS      = d.day.nunique()

bucket = lambda x: pd.cut(x, TENURE_EDGES, labels=TENURE_LABELS)
d['tenure'] = bucket(d.D1)
d['M_rel']  = d.tenure.map(ten.M_relationship).astype(float).fillna(ten.M_relationship.median())

A = d.TransactionAmt.values
d['L'] = A*COGS + CB_FEE + OPS_DISP          # cost of a false negative
d['M'] = A*MARGIN + d.M_rel.values           # cost of a false positive
L, M, p, y = d.L.values, d.M.values, d.p.values, d.isFraud.values

print(f"== TEST WINDOW: {len(d):,} transactions, {DAYS} days, ${A.sum():,.0f} GMV, "
      f"{y.mean():.2%} fraud")
print(f"\nper-transaction FN cost L : median ${np.median(L):.2f}  mean ${L.mean():.2f}  p99 ${np.percentile(L,99):.2f}")
print(f"per-transaction FP cost M : median ${np.median(M):.2f}  mean ${M.mean():.2f}  p99 ${np.percentile(M,99):.2f}")
print(f"L/M ratio                 : median {np.median(L/M):.2f}  p10 {np.percentile(L/M,10):.2f}  p90 {np.percentile(L/M,90):.2f}")

# ---------------- EV of each action, per transaction ----------------
def evs(p, L, M):
    return np.vstack([
        -p*L,                                                   # allow
        -(p*F_PASS*L + (1-p)*A_ABANDON*M),                      # challenge
        -((1-Q_ANALYST)*(p*L + (1-p)*M) + C_REVIEW + D_DELAY),  # review
        -(1-p)*M,                                               # deny
    ])
ACT = np.array(['allow','challenge','review','deny'])

# ---------------- REALISED cost given the true label ----------------
def realised(action_idx, y, L, M):
    c = np.zeros(len(y))
    a0 = action_idx==0; c[a0] = y[a0]*L[a0]
    a1 = action_idx==1; c[a1] = y[a1]*F_PASS*L[a1] + (1-y[a1])*A_ABANDON*M[a1]
    a2 = action_idx==2; c[a2] = (1-Q_ANALYST)*(y[a2]*L[a2] + (1-y[a2])*M[a2]) + C_REVIEW + D_DELAY
    a3 = action_idx==3; c[a3] = (1-y[a3])*M[a3]
    return c

def summarise(name, act, score_used=None):
    c = realised(act, y, L, M)
    mix = pd.Series(act).value_counts(normalize=True).reindex([0,1,2,3]).fillna(0)
    fn = ((act==0) & (y==1)).sum()                      # fraud allowed straight through
    fp = ((act==3) & (y==0)).sum()                      # good customer hard-declined
    caught = y.sum()-fn
    return dict(policy=name, total=c.sum(), annual=c.sum()*365/DAYS,
                per_txn=c.mean(), allow=mix[0], chal=mix[1], rev=mix[2], deny=mix[3],
                n_review=int((act==2).sum()), fraud_through=fn, fraud_caught_pct=caught/y.sum(),
                good_denied=fp, good_denied_pct=fp/(y==0).sum())

res = []

# 1. do nothing
res.append(summarise('P0  approve everything', np.zeros(len(y),dtype=int)))

# 2. binary allow/deny, single global threshold, swept for best P&L
best=None
for t in np.unique(np.quantile(p, np.linspace(0.90,0.9999,300))):
    act = np.where(p>=t, 3, 0)
    c = realised(act,y,L,M).sum()
    if best is None or c<best[1]: best=(t,c,act)
res.append(summarise(f'P1  binary, global thr={best[0]:.4f}', best[2]))
GLOBAL_T = best[0]

# 3. the "we shipped a good model" policy: decline the top 1% by score
t99 = np.quantile(p, 0.99)
res.append(summarise('P2  binary, decline top 1% by score', np.where(p>=t99,3,0)))

# 4. binary but per-transaction threshold (deny iff EV(deny)>EV(allow))
e = evs(p,L,M)
res.append(summarise('P3  binary, per-txn EV threshold', np.where(e[3]>e[0],3,0)))

# 5. full four-action EV argmax
act_ev = e.argmax(axis=0)
res.append(summarise('P4  four-action EV argmax', act_ev))

R = pd.DataFrame(res)
print("\n\n================ POLICY COMPARISON (32-day test window, real labels) ================")
show = R.copy()
for c in ['total','annual']: show[c] = show[c].map(lambda v: f"${v:,.0f}")
for c in ['allow','chal','rev','deny','fraud_caught_pct','good_denied_pct']:
    show[c] = show[c].map(lambda v: f"{v:.2%}")
show['per_txn'] = show.per_txn.map(lambda v: f"${v:.3f}")
print(show[['policy','total','annual','per_txn','allow','chal','rev','deny',
            'fraud_caught_pct','good_denied','n_review']].to_string(index=False))

base = R.loc[0,'annual']
print(f"\nannualised cost vs 'approve everything' (${base:,.0f}/yr):")
for _,r in R.iloc[1:].iterrows():
    print(f"  {r.policy:<38} ${r.annual:,.0f}   saves ${base-r.annual:,.0f}/yr  ({(base-r.annual)/base:.1%})")

# ---------------- per-transaction decision boundaries ----------------
def bnd_allow_chal(L,M): return A_ABANDON*M/(L*(1-F_PASS)+A_ABANDON*M)
def bnd_chal_deny(L,M):  return (1-A_ABANDON)*M/(F_PASS*L+(1-A_ABANDON)*M)
def bnd_allow_deny(L,M): return M/(L+M)
b1, b2, b0 = bnd_allow_chal(L,M), bnd_chal_deny(L,M), bnd_allow_deny(L,M)
print("\n\n================ THE BOUNDARIES ARE A DISTRIBUTION, NOT A NUMBER ================")
q = [1,10,25,50,75,90,99]
print(pd.DataFrame({'allow->challenge':np.percentile(b1,q), 'challenge->deny':np.percentile(b2,q),
                    'allow->deny (binary)':np.percentile(b0,q)}, index=[f'p{x}' for x in q]).round(4))
print(f"\nspread of the binary allow/deny boundary: p1={np.percentile(b0,1):.3f} .. "
      f"p99={np.percentile(b0,99):.3f}  ({np.percentile(b0,99)/np.percentile(b0,1):.1f}x)")

d['b_allow_deny']=b0; d['b_allow_chal']=b1; d['b_chal_deny']=b2
d['act_ev']=act_ev
print("\n== boundary by tenure segment (median)")
print(d.groupby('tenure', observed=True).agg(n=('p','size'), fraud=('isFraud','mean'),
      med_M=('M','median'), med_L=('L','median'),
      med_bound=('b_allow_deny','median'), deny_rate=('act_ev',lambda s:(s==3).mean())).round(4))
print("\n== boundary by amount band (median)")
d['amt_band']=pd.cut(d.TransactionAmt,[0,25,50,100,250,500,1e6],
                     labels=['<25','25-50','50-100','100-250','250-500','500+'])
print(d.groupby('amt_band', observed=True).agg(n=('p','size'), fraud=('isFraud','mean'),
      med_L=('L','median'), med_M=('M','median'),
      med_bound=('b_allow_deny','median'), deny_rate=('act_ev',lambda s:(s==3).mean())).round(4))

d.to_parquet('data/econ_test.parquet')
R.to_csv('data/policies.csv', index=False)
json.dump(dict(GLOBAL_T=float(GLOBAL_T), DAYS=int(DAYS)), open('data/consts.json','w'))
print("\nsaved econ_test.parquet")
