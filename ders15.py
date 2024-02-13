import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt 

data=pd.read_csv("sarap.csv")
veri=data.copy()


y=veri["quality"]

X=veri.drop(columns="quality",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


pca=PCA(n_components=11)

X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)


print(np.cumsum(pca.explained_variance_ratio_)*100)#kümeletik olakar

lm=LinearRegression()
lm.fit(X_train2,y_train)
tahmin=lm.predict(X_test2)

r2=mt.r2_score(y_test,tahmin)
rmse=mt.mean_squared_error(y_test,tahmin,squared=True)

print("R2:{} RMSE:{}".format(r2,rmse))

cv=KFold(n_splits=10,shuffle=True,random_state=1)

lm2=LinearRegression()
RMSE=[]

for i in range(1,X_train2.shape[1]+1):
    hata=np.sqrt(-1*cross_val_score(lm2,X_train2[:,:i],y_train.ravel(),cv=cv,scoring="neg_mean_squared_error").mean())
    RMSE.append(hata)
plt.plot(RMSE,"-x")
plt.xlabel("bileşen sayisi")
plt.ylabel("hatalar")
plt.show()
    