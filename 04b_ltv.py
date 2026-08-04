import pandas as pd, numpy as np
from config import (MARGIN, DISCOUNT, P_CHURN_ON_DECLINE,
                    TENURE_EDGES, TENURE_LABELS, DATA)
"""
Fix in 04: LTV per tenure conditioned on uids with >=2 transactions -- survivorship
bias, worst in the 'new' bucket where most uids never return. Correct it by measuring
P(repeat) directly and taking residual LTV = P(repeat) x value-if-repeat x discount.
"""
m = pd.read_parquet('data/meta.parquet')

# uid-level panel over the whole 182 days
u = m.groupby('uid').agg(spend=('TransactionAmt','sum'), n=('TransactionAmt','size'),
                         first=('day','min'), last=('day','max'), D1first=('D1','first'))
# a uid is only fairly assessed for "did it repeat" if we can watch it for 60+ days
watch = u[u['first'] <= 182-60].copy()
watch['repeated'] = (watch.n >= 2).astype(int)
watch['obs_days'] = (watch['last']-watch['first']+1).clip(lower=1)

def bucket(d):
    return pd.cut(d, TENURE_EDGES, labels=TENURE_LABELS)
watch['tenure'] = bucket(watch.D1first)

rows=[]
for t, g in watch.groupby('tenure', observed=True):
    p_rep = g.repeated.mean()
    rp = g[(g.repeated==1)&(g.obs_days>=7)]
    ann_if_repeat = (rp.spend/rp.obs_days*365).median() if len(rp)>30 else np.nan
    rows.append(dict(tenure=t, n_uid=len(g), p_repeat=p_rep,
                     ann_gmv_if_repeat=ann_if_repeat))
tab = pd.DataFrame(rows).set_index('tenure')
tab['expected_ann_gmv'] = tab.p_repeat*tab.ann_gmv_if_repeat
tab['residual_LTV'] = tab.expected_ann_gmv*MARGIN*DISCOUNT

tab['p_churn'] = pd.Series(tab.index.astype(str)).map(P_CHURN_ON_DECLINE).values
tab['M_relationship'] = tab.p_churn*tab.residual_LTV

fr = m[m.split=='train'].assign(tenure=lambda d: bucket(d.D1)).groupby(
        'tenure', observed=True).agg(n=('isFraud','size'), fraud_rate=('isFraud','mean'),
        med_amt=('TransactionAmt','median'))
tab = fr.join(tab, how='left')

print("== tenure segments, all quantities measured except p_churn\n")
print(tab[['n','fraud_rate','med_amt','p_repeat','ann_gmv_if_repeat',
           'residual_LTV','p_churn','M_relationship']].round(3).to_string())
print(f"\noverall P(repeat) = {watch.repeated.mean():.3f} over {len(watch):,} watchable uids")
tab.to_csv('data/tenure_econ.csv')
print("\nsaved data/tenure_econ.csv")
