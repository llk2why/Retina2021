import cv2
import numpy as np
from utils.util import *

def test_metric():
    im1 = cv2.imread('test/a=0.0000_b=0.0000.png')
    im2 = cv2.imread('test/a=0.0000_b=0.0000.jpg')
    psnr,ssim = calc_metrics(im1,im2,0)
    print(psnr,ssim)


if __name__ == '__main__':
    test_metric()