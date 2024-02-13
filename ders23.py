import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt 
data=pd.read_csv("maas.csv")
veri=data.copy()

y=veri["Salary"]
X=veri["Level"]

y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)

dtmodel=DecisionTreeRegressor(random_state=0)
dtmodel.fit(X,y)
dttahmin=dtmodel.predict(X)

rfmodel=RandomForestRegressor(random_state=0)
rfmodel.fit(X,y)
rftahmin=rfmodel.predict(X)


plt.scatter(X,y,color="red")
plt.plot(X,dttahmin,color="blue")
plt.plot(X,rftahmin,color="green")
plt.show()