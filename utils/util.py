import os
import cv2
import math
import torch
import shutil
import datetime
import numpy as np


""" #################### """
"""     common tools     """
""" #################### """

def get_timestamp(short=False):
    if short:
        return datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths,exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path,exist_ok=True)

def mkdir_and_rename(path):
    ret_name = None
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
        ret_name = new_name
    os.makedirs(path)
    if ret_name is None:
        ret_name = path
    return ret_name


""" #################### """
"""   Image Processing   """
""" #################### """

def quantize(img):
    # print(img.min())
    # print(img.max())
    # exit()
    max_rgb = 255
    return img.mul(max_rgb).clamp(0, max_rgb).round()

def tensorToNumpy(tensor_list):

    def _tensorToNumpy(tensor):
        array = np.transpose(quantize(tensor).numpy(), (1, 2, 0)).astype(np.uint8)
        return array

    return [_tensorToNumpy(tensor) for tensor in tensor_list]


# https://stackoverflow.com/questions/55606636/why-opencv-rgb-to-ycbcr-conversion-doesnt-give-the-same-result-as-conversion-in
def rgb2ycbcr(img, only_y=False):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)

    # convert
    ycbcr = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                            [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    ycbcr = ycbcr.round()
    if only_y:
        ycbcr = ycbcr[...,0]
    return ycbcr.astype(np.uint8)


""" #################### """
"""        metric        """
""" #################### """
def calc_metrics(img1, img2, crop_border=None, test_Y=True):
    if crop_border is None:
        crop_border = 0

    im1_in = rgb2ycbcr(img1,test_Y)
    im2_in = rgb2ycbcr(img2,test_Y)

    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1, cropped_im2)
    ssim = calc_ssim(cropped_im1, cropped_im2)
    return psnr, ssim


def calc_psnr(img1, img2):

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')