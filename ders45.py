import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
data=pd.read_csv("müsteriler.csv")
veri=data.copy()


veri=veri.drop(columns="CustomerID",axis=1)

X=veri.iloc[:,1:3]
# # grafik k değeri öğrenme
# kmodel=KMeans(random_state=0)
# grafik=KElbowVisualizer(kmodel,k=(1,20))
# grafik.fit(X)
# grafik.poof()




# wcss=[]

# for k in range(1,20):
#     kmodel=KMeans(n_clusters=k,random_state=0)
#     kmodel.fit(X)
#     wcss.append(kmodel.inertia_)
# plt.plot(range(1,20),wcss)
# plt.title("puan")
# plt.xlabel("kümeler sayisi") 
# plt.ylabel("wcss")
# plt.show()   
