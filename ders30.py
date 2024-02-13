import pandas as pd 
import matplotlib.pyplot as plt 
data=pd.read_csv("kanser.csv")
veri=data.copy()

M=veri[veri["diagnosis"]=="M"]

B=veri[veri["diagnosis"]=="B"]

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kötü huylu")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi huylu")
plt.legend()
plt.show()