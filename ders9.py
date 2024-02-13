import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as mt
data=pd.read_csv("ev.csv")
veri=data.copy()

veri.drop(columns=["No","X1 transaction date","X5 latitude","X6 longitude"],axis=1,inplace=True)

veri=veri.rename(columns={"X2 house age":"ev yaş",
"X3 distance to the nearest MRT station":"Metro uzaklık",
"X4 number of convenience stores ":"market sayisi",
"Y house price of unit area ":"Ev fiyatı"})
print(veri.isnull())

y=veri["Y house price of unit area"]
X=veri.drop(columns="Y house price of unit area",axis=1)

pol=PolynomialFeatures(degree=3)
X_pol=pol.fit_transform(X)


X_train,X_test,y_train,y_test=train_test_split(X_pol,y,test_size=0.2,random_state=42)




pol_reg=LinearRegression()
pol_reg.fit(X_train,y_train)
tahmin=pol_reg.predict(X_test)


r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)

print("R2: {}  MSE:{}".format(r2,mse))




# sns.pairplot(veri)
# plt.show()
