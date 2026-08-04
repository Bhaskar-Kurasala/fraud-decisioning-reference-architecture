import pandas as pd, numpy as np
from config import DATA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

X = pd.read_parquet('data/X.parquet', columns=['isFraud','split','day'])
split = X.split.values; y = X.isFraud.values.astype(np.int8)
yca, yte = y[split=='calib'], y[split=='test']
p_ca_raw = np.load('data/p_ca_raw.npy'); p_te_raw = np.load('data/p_te_raw.npy')

iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1-1e-6).fit(p_ca_raw, yca)
p_te = iso.predict(p_te_raw)

# Analytic class-rebalanced score: training at a 50/50 effective prior multiplies the
# odds by (1-pi)/pi. This is exactly the artefact a class_weight='balanced' fit produces
# when nobody undoes it -- the ranking is identical, the probabilities are not.
pi = y[split=='train'].mean(); r = (1-pi)/pi
p_bal = (p_te_raw*r) / (p_te_raw*r + (1-p_te_raw))
print(f"train base rate {pi:.4f} -> rebalancing inflates odds by {r:.1f}x")

def ece(y_, p, bins=20):
    e = np.quantile(p, np.linspace(0,1,bins+1)); e[0], e[-1] = -1, 2
    i = np.digitize(p, e[1:-1]); t=0.0
    for b in range(bins):
        m = i==b
        if m.sum(): t += m.mean()*abs(p[m].mean()-y_[m].mean())
    return t

def rep(n, y_, p):
    print(f"{n:<28} AUC {roc_auc_score(y_,p):.4f}  PR-AUC {average_precision_score(y_,p):.4f}"
          f"  Brier {brier_score_loss(y_,p):.5f}  ECE {ece(y_,p):.5f}  mean_p {p.mean():.4f}")

print(f"\n== TEST days 150-181 (out-of-time), actual fraud rate {yte.mean():.4f}, n={len(yte):,}")
rep('champion raw', yte, p_te_raw)
rep('champion + isotonic', yte, p_te)
rep('class-rebalanced (uncorrected)', yte, p_bal)

def rel(y_, p, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates='drop')
    d = pd.DataFrame({'p':p,'y':y_,'b':q}).groupby('b').agg(
        n=('y','size'), predicted=('p','mean'), actual=('y','mean'))
    d['over_under'] = (d.predicted/d.actual.replace(0,np.nan)); return d

for nm,p in [('champion RAW (uncalibrated)',p_te_raw), ('champion + ISOTONIC',p_te),
             ('CLASS-REBALANCED, uncorrected',p_bal)]:
    print(f"\n== reliability decile: {nm}"); print(rel(yte,p).round(4))

meta = pd.read_parquet('data/meta.parquet')
out = meta.loc[split=='test', ['TransactionID','isFraud','TransactionAmt','ProductCD',
                               'day','D1','card6','card4']].copy()
out['p']=p_te; out['p_raw']=p_te_raw; out['p_bal']=p_bal
out.to_parquet('data/scored_test.parquet')
print("\nsaved scored_test.parquet", out.shape)
print("\nscore distribution (calibrated):")
print(pd.Series(p_te).describe(percentiles=[.5,.9,.95,.99,.999]).round(5))
