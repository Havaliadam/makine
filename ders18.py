import yfinance as yf 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt 
import sklearn.metrics as mt 
data=yf.download("THYAO.IS",start="2022-08-01",end="2022-09-01")
veri=data.copy()
# print(veri.columns)
veri=veri.reset_index()
veri["Day"]=veri["Date"].astype(str).str.split("-").str[2]

y=veri["Adj Close"]
X=veri["Day"]

y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)


scy=StandardScaler()
scx=StandardScaler()

X=scx.fit_transform(X)
y=scy.fit_transform(y)


svrrbf=SVR(kernel="rbf",C=10000)
svrrbf.fit(X,y)
tahminrbf=svrrbf.predict(X)



# svrlin=SVR(kernel="linear")
# svrlin.fit(X,y)
# tahminlin=svrlin.predict(X)


# svrpoly=SVR(kernel="poly",degree=5)
# svrpoly.fit(X,y)
# tahminpoly=svrpoly.predict(X)


r2=mt.r2_score(y,tahminrbf)
rmse=mt.mean_squared_error(y,tahminrbf,squared=False)
print("R2:{}  RMSE:{}".format(r2,rmse)
      )




plt.scatter(X,y,color="red")
plt.plot(X,tahminrbf,color="green",label="RBF Model ")
# plt.plot(X,tahminlin,color="blue",label="Linear Model ")
# plt.plot(X,tahminpoly,color="black",label="Poly  Model ")
plt.legend()
plt.show()

