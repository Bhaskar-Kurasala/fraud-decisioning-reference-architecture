import pandas as pd, numpy as np, gc, json
from config import TRAIN_END, CALIB_END, RAW_TXN, RAW_ID, DATA

TR, ID = str(RAW_TXN), str(RAW_ID)

base = ['TransactionID','isFraud','TransactionDT','TransactionAmt','ProductCD',
        'card1','card2','card3','card4','card5','card6','addr1','addr2',
        'dist1','dist2','P_emaildomain','R_emaildomain']
Ccols = [f'C{i}' for i in range(1,15)]
Dcols = [f'D{i}' for i in range(1,16)]
Mcols = [f'M{i}' for i in range(1,10)]
# V columns: use the subset the community found most informative (low-null, high-signal)
Vkeep = [1,3,4,6,8,11,13,14,17,20,23,26,27,30,36,37,40,41,44,47,48,54,56,59,62,65,
         67,68,70,76,78,80,82,86,88,89,91,107,108,111,115,117,120,121,123,124,127,
         129,130,136,138,139,142,147,156,162,165,160,166,178,176,173,182,187,203,
         205,207,215,169,171,175,180,185,188,198,210,209,218,223,224,226,228,229,
         235,240,258,257,253,252,260,261,264,266,267,274,277,220,221,234,238,250,
         271,294,284,285,286,291,297,303,305,307,309,310,320,281,283,289,296,301,
         314,332,325,335,338]
Vcols = [f'V{i}' for i in Vkeep]

df = pd.read_csv(TR, usecols=base+Ccols+Dcols+Mcols+Vcols)
idf = pd.read_csv(ID, usecols=['TransactionID','id_01','id_02','id_05','id_06','id_09',
                               'id_11','id_13','id_14','id_17','id_19','id_20',
                               'id_30','id_31','id_33','id_38','DeviceType','DeviceInfo'])
df = df.merge(idf, on='TransactionID', how='left'); del idf; gc.collect()
print("merged", df.shape)

t0 = df.TransactionDT.min()
df['day'] = ((df.TransactionDT - t0)//86400).astype(int)
df['hour'] = ((df.TransactionDT//3600) % 24).astype(int)
df['dow']  = ((df.TransactionDT//86400) % 7).astype(int)
df['logAmt'] = np.log1p(df.TransactionAmt)
df['amt_cents'] = ((df.TransactionAmt - df.TransactionAmt.astype(int))*100).round().astype(int)
df['amt_round'] = (df.amt_cents == 0).astype(int)

# ---- entity keys (classic IEEE-CIS uid construction) ----
df['uid'] = (df.card1.astype(str) + '_' + df.card2.astype(str) + '_' +
             df.addr1.astype(str) + '_' + (df.D1 - df.day).round().astype(str))
df['dev'] = df.DeviceInfo.astype(str) + '_' + df.id_31.astype(str)

# ---------------- OUT-OF-TIME SPLIT ----------------
# train: days 0-119 | calib: 120-149 | test: 150-181
TR_END, CAL_END = TRAIN_END, CALIB_END
split = np.where(df.day < TR_END, 'train', np.where(df.day < CAL_END, 'calib', 'test'))
df['split'] = split
print(df.groupby('split').agg(n=('isFraud','size'), fraud=('isFraud','sum'),
      rate=('isFraud','mean'), days=('day', lambda s:(s.min(),s.max()))))

trmask = df.split == 'train'

# ---- frequency encodings FIT ON TRAIN ONLY (no leakage) ----
for col in ['card1','card2','card3','card5','addr1','P_emaildomain','R_emaildomain',
            'uid','dev','DeviceInfo','id_31','id_19','id_20']:
    vc = df.loc[trmask, col].value_counts()
    df[f'{col}_cnt'] = df[col].map(vc).fillna(0).astype(np.float32)

# ---- amount-relative-to-entity aggregates, stats from TRAIN only ----
for key in ['card1','uid','P_emaildomain']:
    g = df.loc[trmask].groupby(key).TransactionAmt.agg(['mean','std'])
    df[f'amt_over_{key}_mean'] = (df.TransactionAmt / df[key].map(g['mean'])).astype(np.float32)
    df[f'amt_z_{key}']         = ((df.TransactionAmt - df[key].map(g['mean'])) /
                                  df[key].map(g['std']).replace(0,np.nan)).astype(np.float32)

# ---- historical velocity: strictly backward-looking, no leakage by construction ----
df = df.sort_values('TransactionDT').reset_index(drop=True)
for key in ['uid','card1']:
    g = df.groupby(key)
    df[f'{key}_txn_idx']  = g.cumcount().astype(np.float32)
    df[f'{key}_secs_prev'] = (df.TransactionDT - g.TransactionDT.shift(1)).astype(np.float32)
    df[f'{key}_amt_prev']  = g.TransactionAmt.shift(1).astype(np.float32)
    df[f'{key}_amt_cummean'] = (g.TransactionAmt.cumsum() - df.TransactionAmt) / \
                               df[f'{key}_txn_idx'].replace(0,np.nan)

cats = ['ProductCD','card4','card6','P_emaildomain','R_emaildomain','M1','M2','M3','M4',
        'M5','M6','M7','M8','M9','id_30','id_31','id_33','id_38','DeviceType','DeviceInfo']
for c in cats:
    df[c] = df[c].astype('category')

drop = ['TransactionID','isFraud','TransactionDT','split','uid','dev','day']
feats = [c for c in df.columns if c not in drop]
print(f"\n{len(feats)} features")

df[['TransactionID','isFraud','TransactionDT','TransactionAmt','ProductCD','day',
    'split','D1','card6','card4','uid']].to_parquet('data/meta.parquet')
df[feats + ['isFraud','split','day']].to_parquet('data/X.parquet')
json.dump(feats, open('data/feats.json','w'))
print("saved")
