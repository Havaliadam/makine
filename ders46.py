import pandas as pd 
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage
data=pd.read_csv("müsteriler.csv")
veri=data.copy()


X=veri.iloc[:,2:4]



model=AgglomerativeClustering()
tahmin=model.fit_predict(X)

X["Label"]=tahmin
# print(X)
# print(X["Age"][X["Label"]==0])
print(X["Age"][X["Label"]==1])

plt.scatter(X["Age"][X["Label"]==0],X["Annual Income (k$)"][X["Label"]==0],c="red")
plt.scatter(X["Age"][X["Label"]==1],X["Annual Income (k$)"][X["Label"]==1],c="black")
plt.show()

link=linkage(X)
dendrogram(link)
plt.xlabel("veri noktaları")
plt.ylabel("mesafe")
plt.show()
