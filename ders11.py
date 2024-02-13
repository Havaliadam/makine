import pandas as pd 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,LassoCV

import numpy as np 
df=load_diabetes()

data=pd.DataFrame(df.data,columns=df.feature_names)
veri=data.copy()

veri["health"]=df.target

#print(veri)

y=veri["health"]
X=veri.drop(columns="health",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# ridge_model=Ridge(alpha=0.1)
# ridge_model.fit(X_train,y_train)

#tahmin=ridge_model.predict(X_test)

# print(ridge_model.score(X_train,y_train))#başarı puanı
# print(ridge_model.score(X_test,y_test))# test dataset puanlama
#print(mt.r2_score(y_test,tahmin))# test dataset puanlama yaoar

lasso_model=Lasso(alpha=0.01)

lasso_model.fit(X_train,y_train)

print(lasso_model.score(X_train,y_train))#lasso değerler
print(lasso_model.score(X_test,y_test))

lamb=LassoCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_


lasso_model2=Lasso(alpha=lamb)

lasso_model2.fit(X_train,y_train)

print(lasso_model2.score(X_train,y_train))#lasso değerler
print(lasso_model2.score(X_test,y_test))



# print(ridge_model.coef_)
# print(lasso_model.coef_)


# print(np.sum(ridge_model.coef_!=0))
# print(np.sum(lasso_model.coef_!=0))