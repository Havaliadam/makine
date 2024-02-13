import numpy as np 
import cv2 
from skimage.measure import block_reduce
resim=cv2.imread("kus.jpg")
resim=cv2.cvtColor(resim,cv2.COLOR_BGR2GRAY)

filtre=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
filtre_resim=cv2.filter2D(resim,-1,filtre)
relu=np.maximum(filtre_resim,0)

pool_size=(2,2)
pool_resim=block_reduce(relu,pool_size,np.max)

cv2.imshow("orjinal resim",resim)
cv2.imshow("relu resim",relu)
cv2.imshow("pool resim",pool_resim)
cv2.waitKey(0)
