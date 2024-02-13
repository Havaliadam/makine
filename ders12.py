import pandas as pd 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet,Ridge,Lasso,ElasticNetCV
import sklearn.metrics as mt 

df=load_diabetes()

data=pd.DataFrame(df.data,columns=df.feature_names)
veri=data.copy()

veri["health"]=df.target


y=veri["health"]
X=veri.drop(columns="health",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)


lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)



elastic_model=ElasticNet(alpha=0.1)
elastic_model.fit(X_train,y_train)

# print(ridge_model.score(X_train,y_train))
# print(lasso_model.score(X_train,y_train))
print(elastic_model.score(X_train,y_train))



# print(ridge_model.score(X_test,y_test))
# print(lasso_model.score(X_test,y_test))
print(elastic_model.score(X_test,y_test))


# tahminid=ridge_model.predict(X_test)
# tahminlass=lasso_model.predict(X_test)
tahminelas=elastic_model.predict(X_test)


# print(mt.mean_squared_error(y_test,tahminid))
# print(mt.mean_squared_error(y_test,tahminlass))
print(mt.mean_squared_error(y_test,tahminelas))

lamb=ElasticNetCV(cv=10,max_ter=10000).fit(X_train,y_train).alpha_

elastic_model2=ElasticNet(alpha=lamb)
elastic_model2.fit(X_train,y_train)



print(elastic_model2.score(X_train,y_train))
print(elastic_model2.score(X_test,y_test))


tahminelas2=elastic_model2.predict(X_test)
print(mt.mean_squared_error(y_test,tahminelas2))




