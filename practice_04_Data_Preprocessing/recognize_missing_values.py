import pandas as pd
import numpy as np
from io import StringIO
# =============================================================================
# for padding missing values
# =============================================================================
from sklearn.impute import SimpleImputer

csv_data = \
"""
A, B, C, D
1.0,2.0,3.0,4.0
5.0,6.0,, 8.0
10.0,11.0,12.0,
"""

df = pd.read_csv(StringIO(csv_data))
print(f"Raw data:\n{df}", '\n')
#print(df.isnull(), '\n')
#print(df.isnull().sum(), '\n')

''' Delete missing values '''
tmp_df = df.dropna(axis=0)
print(f"Delete row(s) if missing:\n{tmp_df}", '\n')
tmp_df = df.dropna(axis=1)
print(f"Delete column(s) if missing:\n{tmp_df}", '\n')
tmp_df = df.dropna(how="all")
print(f"Delete column(s) if missing:\n{tmp_df}", '\n')

''' Padding missing values '''
# method 1:
cols = df.columns
imr = SimpleImputer(missing_values=np.nan, strategy="mean")
imr = imr.fit(df.values)
imputed_data = pd.DataFrame(imr.transform(df.values), columns=cols)
print(f"Impured data (method 1):\n{imputed_data}", '\n')

# method 2:
tmp_df = df.fillna(df.mean())
print(f"Impured data (method 2):\n{tmp_df}", '\n')
