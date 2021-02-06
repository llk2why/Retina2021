import cv2
import numpy as np


im = np.load('BIND41_RTN100_16.npy')
cv2.imwrite('BIND41_RTN100_16.png',im*255)