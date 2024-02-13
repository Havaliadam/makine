import numpy as np 
from scipy.signal import convolve2d
import matplotlib.pyplot as plt 

input=np.array([[0,1,0,0],[1,1,0,0],[1,0,1,0],[1,1,1,0],[1,0,0,0]])
filtre=np.array([[1,0,1],[0,1,0],[1,0,1]])

out=convolve2d(input,filtre,mode="valid")

fig,ax=plt.subplots(1,2)

ax[0].imshow(input,cmap="gray")
ax[1].imshow(out,cmap="gray")
plt.show()