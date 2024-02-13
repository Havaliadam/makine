import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
data=pd.read_csv("maas1.csv")
veri=data.copy()


y=veri["Salary"]
x=veri["YearsExperience"]

# plt.scatter(x,y)
# plt.show()

sabit=sm.add_constant(x)
model=sm.OLS(y,sabit).fit()
#print(model.summary())


lr=LinearRegression()
lr.fit(x.values.reshape(-1,1),y.values.reshape(-1,1))
print(lr.predict(x.values.reshape(-1,1)))







