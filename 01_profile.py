import pandas as pd, numpy as np, json
pd.set_option('display.width', 200)

usecols = ['TransactionID','isFraud','TransactionDT','TransactionAmt','ProductCD',
           'card1','card2','card3','card4','card5','card6','addr1','addr2',
           'dist1','dist2','P_emaildomain','R_emaildomain',
           'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
           'D1','D2','D3','D4','D5','D10','D11','D15','V257','V258','V294','V317']
df = pd.read_csv('data/train_transaction.csv', usecols=usecols)
print("rows", len(df), "cols", df.shape[1])

n = len(df); k = df.isFraud.sum()
print(f"\n== BASE RATE\nfraud {k:,} / {n:,} = {k/n:.4%}")

# temporal
dt = df.TransactionDT
print(f"\n== TIME\nDT span {dt.min():,} .. {dt.max():,} seconds = {(dt.max()-dt.min())/86400:.1f} days")
df['day'] = (df.TransactionDT - dt.min()) // 86400
daily = df.groupby('day').agg(n=('isFraud','size'), fr=('isFraud','mean'),
                              amt=('TransactionAmt','sum'))
print(daily.describe().round(4))
print("\nfraud rate by month-ish (30d block):")
df['block'] = df.day // 30
print(df.groupby('block').agg(n=('isFraud','size'), fraud=('isFraud','sum'),
      rate=('isFraud','mean'), med_amt=('TransactionAmt','median')).round(4))

# amount
a = df.TransactionAmt
print(f"\n== AMOUNT (USD)\n{a.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2)}")
print(f"total volume ${a.sum():,.0f}")
print(f"\nfraud amount vs legit:")
print(df.groupby('isFraud').TransactionAmt.describe(percentiles=[.5,.9,.99]).round(2))
print(f"\n$ at risk (sum of fraud amt) = ${df.loc[df.isFraud==1,'TransactionAmt'].sum():,.0f}"
      f"  = {df.loc[df.isFraud==1,'TransactionAmt'].sum()/a.sum():.3%} of volume")

# amount decile fraud rate -> is exposure correlated with risk?
df['amt_dec'] = pd.qcut(a, 10, labels=False, duplicates='drop')
print("\n== fraud rate by amount decile")
print(df.groupby('amt_dec').agg(n=('isFraud','size'), rate=('isFraud','mean'),
      lo=('TransactionAmt','min'), hi=('TransactionAmt','max')).round(4))

# segments
print("\n== ProductCD")
print(df.groupby('ProductCD').agg(n=('isFraud','size'), rate=('isFraud','mean'),
      med_amt=('TransactionAmt','median'), tot=('TransactionAmt','sum')).round(4))
print("\n== card4 / card6")
print(df.groupby('card4').agg(n=('isFraud','size'), rate=('isFraud','mean')).round(4))
print(df.groupby('card6').agg(n=('isFraud','size'), rate=('isFraud','mean')).round(4))

# D1 = days since card first seen -> tenure proxy!
print("\n== D1 (days since first transaction on card) as TENURE proxy")
df['tenure_bucket'] = pd.cut(df.D1, [-1,0,7,30,90,180,400,10000],
    labels=['0 (new)','1-7d','8-30d','31-90d','91-180d','181-400d','400d+'])
print(df.groupby('tenure_bucket', observed=True).agg(
    n=('isFraud','size'), rate=('isFraud','mean'), med_amt=('TransactionAmt','median')).round(4))
print("D1 null frac:", df.D1.isna().mean().round(4))

df[['TransactionID','isFraud','TransactionDT','TransactionAmt','ProductCD','day']].to_parquet('data/core.parquet')
print("\nsaved core.parquet")
