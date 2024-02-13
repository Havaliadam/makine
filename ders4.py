import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.api as sn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv("reklam.csv")
veri=data.copy()



print(veri.corr()["Sales"])

Q1=veri["Newspaper"].quantile(0.25)
Q3=veri["Newspaper"].quantile(0.75)
IQR=Q3-Q1
ustsinir=Q3+1.5*IQR
aykiri=veri["Newspaper"]>ustsinir
veri.loc[aykiri,"Newspaper"]=ustsinir


y=veri["Sales"]
x=veri[["TV","Radio","Newspaper"]]

sabit=sn.add_constant(x)
model=sn.OLS(y,sabit).fit()


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin=lr.predict(x_test)
y_test=y_test.sort_index()
# print(tahmin)
df=pd.DataFrame({"Gerçek":y_test,"Tahmin":tahmin})
df.plot(kind="line")
plt.show()
# print(lr.coef_)







# sns.boxplot(veri["Newspaper"])
# plt.show()

# sns.pairplot(veri,kind="reg") 
# plt.show()
