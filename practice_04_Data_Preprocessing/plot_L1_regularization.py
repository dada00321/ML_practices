import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

''' 1. Prepare wine data (13 features, 178 records) '''
csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

df = pd.read_csv(csv_url, header=None)
#print(df.shape) # (178, 14)
feature_names = \
["Alcohol",
 "Malic acid",
 "Ash",
 "Alcalinity of ash",
 "Magnesium",
 "Total phenols",
 "Flavanoids",
 "Nonflavanoid phenols",
 "Proanthocyanins",
 "Color intensity",
 "Hue",
 "OD280/OD315 of diluted wines",
 "Proline"]
df.columns = ["Class label"] + feature_names

fig = plt.figure()
ax = plt.subplot(111)
    
feature_colors = ['blue', 'green', 'red', 'cyan', 
         'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

''' 2. Features preprocessing '''
label_col_name = "Class label"
test_dataset_ratio = 0.3

X = df.loc[:, df.columns != label_col_name].values
Y = df.loc[:, label_col_name].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=1, stratify=Y)
ss = StandardScaler()
ss.fit(X_train) # calculate statistics for normalization
X_train, X_test = ss.transform(X_train), ss.transform(X_test)

''' 3. Train logistic regression model '''
weights, params = list(), list()
base = 10.0
for c in np.arange(-4, 6):
	# If use default solver: `lbfgs`, error will occur like:
	# ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=base**c, random_state=0)
    lr.fit(X_train, Y_train)
    weights.append(lr.coef_[1])
    params.append(base**c)

weights = np.array(weights)

''' 4. Plot costs based on different 
    inverse regularization parameter: C '''
for column, color in zip(range(weights.shape[1]), feature_colors):
    plt.plot(params, weights[:, column],
             label=df.columns[column+1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel("weight coefficient")
plt.xlabel('C')
plt.xscale("log")
plt.legend(loc="upper left")
ax.legend(loc="upper center", 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.savefig("./res/L1_regularization/"+\
			"sklearn_L1_regularization")
plt.show()