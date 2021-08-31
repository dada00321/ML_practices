import pandas as pd
import numpy as np

# =============================================================================
# Encode labels
# =============================================================================
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# Encode nomial features
# =============================================================================
from sklearn.preprocessing import OneHotEncoder

'''
 Categorical > "Ordinal" feature pre-processing
'''
apparel_data = [["green", 'M', 10.1, "T-shirt"],
				["red", 'L', 13.5, "T-shirt"],
				["blue", "XL", 15.3, "jacket"],
				["blue", "S", 6.3, "jacket"]]

column_names = ["color", "size", "price", "class_label"]
label_col_name = column_names[-1]
df = pd.DataFrame(apparel_data)
df.columns = column_names
print(f"Raw data:\n {df}\n")

size_mapping = {"XL":3, 'L':2, 'M':1, 'S':0}
df["size"] = df["size"].map(size_mapping)
print(f"After size mapping:\n {df}\n")

'''
 Label pre-processing
'''
class_mapping = {label: idx
				 for idx, label in 
				     enumerate(np.unique(df[label_col_name]))}
inv_class_mapping = {v: k for k, v in class_mapping.items()}

df[label_col_name] = df[label_col_name].map(class_mapping)
print(f"After label binarizing:\n {df}\n")

df[label_col_name] = df[label_col_name].map(inv_class_mapping)
print(f"Before label binarizing:\n {df}\n")

class_le = LabelEncoder()
transformed_Y = class_le.fit_transform(df[label_col_name].values)
print(f"transformed_Y: {transformed_Y}\n")
Y = class_le.inverse_transform(transformed_Y)
print(f"Y: {Y}\n")

#color_onehot = OneHotEncoder().fit_transform(df.iloc[:, 0].values.reshape(-1,1)).toarray()
color_onehot = OneHotEncoder().fit_transform(df.loc[:, "color"].values.reshape(-1,1)).toarray()
print(f"color_onehot:\n{color_onehot}\n")

