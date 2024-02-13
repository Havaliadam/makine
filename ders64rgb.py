import numpy as np 
import matplotlib.pyplot as plt 

R=np.array([[0,255,0],[255,0,255],[0,255,0]])
G=np.array([[0,0,255],[255,0,255],[255,255,0]])
B=np.array([[0,0,0],[255,255,0],[0,255,255]])



tensör=np.stack([R,G,B],axis=2)

plt.imshow(tensör)
plt.show()