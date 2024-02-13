import numpy as np 
import cv2


resim=cv2.imread("resim2.jpg")
girismat=np.asarray(resim)
filtre=cv2.Laplacian(resim,cv2.CV_64F)
relu=np.maximum(filtre,0)
print(relu)




# cv2.imshow("orjinal resim",resim)
# cv2.imshow("filtre resim",filtre)
# cv2.waitKey(0)









# cv2.imshow("ornek resim",resim)
# cv2.waitKey(0)