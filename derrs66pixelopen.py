import numpy as np 
import cv2
import matplotlib.pyplot as plt 
from scipy.signal import convolve2d

resim=cv2.imread("ornek.jpg")
resim=cv2.cvtColor(resim,cv2.COLOR_BGR2RGB)
resimgri=cv2.cvtColor(resim,cv2.COLOR_BGR2GRAY)

inputmat=np.asarray(resim)
inputmat2=np.asarray(resimgri)



prewittX=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittY=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])


sobelX=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobelY=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])





out=convolve2d(inputmat2,prewittX)
out2=convolve2d(inputmat2,prewittY)


out3=convolve2d(inputmat2,sobelX)
out4=convolve2d(inputmat2,sobelY)


fig,ax=plt.subplots(1,4)
ax[0].imshow(out,cmap="gray")
ax[0].set_title("prewittX resim")

ax[1].imshow(out2,cmap="gray")
ax[1].set_title("prewittY resim")


ax[2].imshow(out3,cmap="gray")
ax[2].set_title("sobel x  resim")

ax[3].imshow(out4,cmap="gray")
ax[3].set_title("sobely  fitrle resim")



plt.show()

#print(inputmat.shape) #matris bilgileri verir  shape 3 kanal yapısını gçsterir
