import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt 

data=pd.read_csv("maas.csv")
veri=data.copy()

y=veri["Salary"]
X=veri["Level"]

y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)

dtr=DecisionTreeRegressor(random_state=0,max_leaf_nodes=5)
dtr.fit(X,y)
tahmin=dtr.predict(X)
print(veri)
print(np.mean(y[:8]))


plt.figure(figsize=(20,10),dpi=100)
plot_tree(dtr,feature_names=X,class_names=y,rounded=True,filled=True)
plt.show()
# plt.scatter(X,y,color="red")
# plt.plot(X,tahmin)

# plt.show()

