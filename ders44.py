import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans

data=pd.read_csv("müsteriler.csv")
veri=data.copy()


veri=veri.drop(columns="CustomerID",axis=1)

#describe hesaplama count mean std min max

X=veri.iloc[:,1:3]




kmodel=KMeans(n_clusters=4,random_state=0)
kfit=kmodel.fit(X)
kumeler=kfit.labels_
merkezler=kfit.cluster_centers_

figure,axis=plt.subplots(1,2)
axis[0].scatter(X.iloc[:,0],X.iloc[:,1],color="black")
# bir grafik olarak parçalama

axis[1].scatter(X.iloc[:,0],X.iloc[:,1],c=kumeler,cmap="winter")
axis[1].scatter(merkezler[:,0],merkezler[:,1],c="red",s=200)
plt.show()


