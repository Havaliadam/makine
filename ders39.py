import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
data=pd.read_csv("diyabet.csv")
veri=data.copy()
y=veri["Outcome"]
X=veri.drop(columns="Outcome",axis=1)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=RandomForestClassifier(random_state=0,criterion="entropy",max_depth=5,min_samples_split=5,n_estimators=1000)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)

acs=accuracy_score(y_test,tahmin)
print(acs*100)

önem=pd.DataFrame({"Önem":model.feature_importances_},index=X.columns)
önem.sort_values(by="Önem",axis=0,ascending=True).plot(kind="barh",color="blue")
plt.title("Değişken önem seviyeleri")
plt.show()



# parametreler={"criterion":["gini","entropy"],"max_depth":[2,5,10],"min_samples_split":[2,5,10],"n_estimators":[50,200,500,1000]}

# grid=GridSearchCV(model,param_grid=parametreler,cv=10,n_jobs=1)
# grid.fit(X_train,y_train)
# print(grid.best_params_)



