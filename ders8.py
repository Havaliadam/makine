import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_excel("deneme2.xlsx")
veri=data.copy()
print(veri)

y=veri["Verim"]
x=veri["Sıcaklık"]


y=y.values.reshape(-1,1)
x=x.values.reshape(-1,1)

lr=LinearRegression()


lr.fit(x,y)
tahmin=lr.predict(x)


r2dog=mt.r2_score(y,tahmin)
mse=mt.mean_squared_error(y,tahmin)

print("dogrusal R2:{} dogrusal mse:{}".format(r2dog,mse))

pol=PolynomialFeatures(degree=3)
x_pol=pol.fit_transform(x)

lr2=LinearRegression()
lr2.fit(x_pol,y)

tahmin2=lr2.predict(x_pol)



r2pol=mt.r2_score(y,tahmin2)
msepol=mt.mean_squared_error(y,tahmin2)

print("polinom R2:{} polinom mse:{}".format(r2pol,msepol))


plt.scatter(x,y,color="red")
plt.plot(x,tahmin2,color="blue")
plt.show()
