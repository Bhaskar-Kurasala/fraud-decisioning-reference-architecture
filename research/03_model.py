import pandas as pd, numpy as np, json, time, gc, os
from config import MODEL, MAX_CAT_LEVELS, DATA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

feats = json.load(open('data/feats.json'))
X = pd.read_parquet('data/X.parquet')
split = X['split'].values; y = X['isFraud'].values.astype(np.int8)

for c in feats:
    if str(X[c].dtype) == 'category':
        if X[c].nunique() > MAX_CAT_LEVELS:
            top = X.loc[split=='train', c].value_counts().head(120).index
            X[c] = X[c].astype(str).where(X[c].isin(top), 'other')
        X[c] = X[c].astype('category')
    else:
        X[c] = pd.to_numeric(X[c], downcast='float')
X = X[feats]; gc.collect()
print("matrix", X.shape, f"{X.memory_usage(deep=True).sum()/1e6:.0f} MB")

tr, ca, te = split=='train', split=='calib', split=='test'
Xtr, ytr = X.loc[tr], y[tr]
Xca, yca = X.loc[ca], y[ca]
Xte, yte = X.loc[te], y[te]
del X; gc.collect()

def fit(w, tag):
    t=time.time()
    m = HistGradientBoostingClassifier(**{**MODEL, "class_weight": w})
    m.fit(Xtr, ytr); print(f"  [{tag}] {m.n_iter_} iters, {time.time()-t:.0f}s")
    return m

m = fit(None, 'champion')
p_ca_raw = m.predict_proba(Xca)[:,1]; p_te_raw = m.predict_proba(Xte)[:,1]
np.save(DATA/'p_te_raw.npy', p_te_raw); np.save(DATA/'p_ca_raw.npy', p_ca_raw)
del m; gc.collect()

# The class-rebalanced variant is derived ANALYTICALLY in 03b (prior shift has a
# closed form and is exactly reproducible). Refitting it is an optional check
# that needs ~6GB; it OOMs on a 3GB box. Enable with FIT_BALANCED=1.
if os.environ.get("FIT_BALANCED") == "1":
    mb = fit('balanced', 'class-balanced')
    np.save(DATA/'p_te_bal_fitted.npy', mb.predict_proba(Xte)[:,1])
    del mb; gc.collect()
del Xtr, Xca, Xte; gc.collect()
print("done -- run 03b_calibrate.py next")
