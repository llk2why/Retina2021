import os
import cv2
import random
import numpy as np

from math import ceil

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF']
BINARY_EXTENSIONS = ['.npy']
BENCHMARK = ['Kodak', 'McM', 'moire','MSR','MIT','vdp','Sandwich']


BIND_PATTERN = {
    'BIND41_RTN100_16':np.array([
        [ 0, 0, 0, 0, 0, 0, 2,-1, 0, 0, 3, 0, 0, 3, 0, 0],
        [ 3, 0, 2,-1, 0, 3, 0, 3, 3, 0,-1, 0, 3,-1, 0, 3],
        [-1, 2,-1, 0, 3,-1, 0,-1,-1, 2,-1, 0,-1, 0, 3,-1],
        [ 2,-1, 0, 0,-1, 0, 0, 0, 2,-1, 0, 0, 0, 0,-1, 0],
        [ 0, 3, 0, 0, 3, 0, 2,-1, 2,-1, 0, 0, 0, 0, 0, 3],
        [ 3,-1, 0, 0,-1, 0, 0, 0, 0, 2,-1, 0, 2,-1, 3,-1],
        [-1, 2,-1, 0, 2,-1, 3, 0, 0, 0, 3, 0, 0, 0,-1, 0],
        [ 2,-1, 0, 0, 0, 0,-1, 0, 2,-1,-1, 0, 0, 0, 2,-1],
        [ 0, 0, 2,-1, 0, 0, 2,-1, 0, 2,-1, 3, 0, 3, 0, 3],
        [ 3, 0, 0, 3, 0, 0, 2,-1, 3, 0, 3,-1, 0,-1, 0,-1],
        [-1, 2,-1,-1, 0, 0, 2,-1,-1, 0,-1, 0, 0, 3, 3, 0],
        [ 0, 0, 0, 0, 0, 0, 2,-1, 0, 0, 0, 0, 0,-1,-1, 0],
        [ 3, 0, 0, 0, 3, 3, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0],
        [-1, 2,-1, 0,-1,-1, 0, 0, 3, 0,-1, 0,-1, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 3, 3,-1, 0, 0, 0, 3, 3, 0, 3],
        [ 2,-1, 2,-1, 0, 0,-1,-1, 2,-1, 2,-1,-1,-1, 0,-1],
    ])
}



""" #################### """
"""     FIles & IO       """
""" #################### """
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS) 


def _get_paths_from_images(path):
    assert os.path.isdir(path), '[Error] {} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path,followlinks=True)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{} has no valid image file'.format(path)
    return images


def get_image_paths(datadir):
    paths = None
    if datadir is not None:
        paths = sorted(_get_paths_from_images(datadir))
    return paths


def find_benchmark(datadir):
    bm_list = [datadir.find(bm)>=0 for bm in BENCHMARK]
    if sum(bm_list) > 0:
        bm_idx = bm_list.index(True)
        bm_name = BENCHMARK[bm_idx]
    else:
        bm_name = 'MyImage'
    return bm_name


def read_img(path):
    ''' read image by opencv
        output: 
            Numpy float32, HWC, RGB, [0,255]
    '''
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def load_cfa(cfa):
    ''' read BGR format cfa
        output: 
            Numpy uint8, HWC, RGB, {0,1}
    '''
    if cfa in ['RGGB','2JCS','3JCS','4JCS','BIND41_RTN100_16']:
        mask = np.load('cfa_pattern/{}.npy'.format(cfa))
    elif cfa in ['Random_pixel','Random_2JCS','Random_3JCS','Random_4JCS','Random_6JCS']:
        base_pattern = np.load('cfa_pattern/random_base128.npy')
        bind_pattern = np.load('cfa_pattern/{}.npy'.format(cfa))
        mask = base_pattern*((bind_pattern!=0)[:,:,None])
    elif cfa == 'Random_base':
        mask = np.load('cfa_pattern/random_base128.npy')
    else:
        raise ValueError('unexpected cfa pattern')
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
    return mask


""" #################### """
"""   Image Processing   """
"""    on Numpy Image    """
""" #################### """
def gen_cfa_with_black(h,w,ratio):
    n = h*w
    nbla = round(n*ratio)
    n_ = n-nbla
    nr,nb = round(n_/4),round(n_/4)
    ng = n_-nr-nb
    assert n == nbla+nr+ng+nb
    seq = [-1]*nbla+[0]*nr+[1]*ng+[2]*nb
    np.random.shuffle(seq)
    flatten_mask = np.array(seq).reshape(h,w)
    ones = np.ones((h,w,3))
    mask = np.stack([np.where(flatten_mask == i, ones[:, :, i], 0) for i in range(3)], axis=-1)
    return mask


def get_cfa(cfa,size=None):
    mask = None
    if cfa in ['RGGB','2JCS','3JCS','4JCS','BIND41_RTN100_16',
               'Random_pixel','Random_2JCS','Random_3JCS',
               'Random_4JCS','Random_6JCS','Random_base']:
        mask = load_cfa(cfa)
    elif cfa == 'Random':
        if size is None:
            raise ValueError('Random pattern without size!')
        if len(size)==3:
            size = size[:2]
        assert len(size) == 2, 'Invalid parameter for size'
        h,w = size
        flatten_mask = np.random.randint(0,3,size=(h,w))
        ones = np.ones((h,w,3))
        mask = np.stack([np.where(flatten_mask == i, ones[:, :, i], 0) for i in range(3)], axis=-1)
    elif cfa == 'Random_RandomBlack20':
        p = np.random.rand()
        if size is None:
            raise ValueError('Random pattern without size!')
        if len(size)==3:
            size = size[:2]
        assert len(size) == 2, 'Invalid parameter for size'
        h,w = size
        if p<0.8:
            flatten_mask = np.random.randint(0,3,size=(h,w))
            ones = np.ones((h,w,3))
            mask = np.stack([np.where(flatten_mask == i, ones[:, :, i], 0) for i in range(3)], axis=-1)
        else:
            mask = gen_cfa_with_black(h,w,0.2)

    elif cfa == 'Random16':
        flatten_mask = np.random.randint(0,3,size=(16,16))
        ones = np.ones((16,16,3))
        mask = np.stack([np.where(flatten_mask == i, ones[:, :, i], 0) for i in range(3)], axis=-1)
    elif cfa == 'RandomBlack20':
        if size is None:
            raise ValueError('Random pattern without size!')
        if len(size)==3:
            size = size[:2]
        assert len(size) == 2, 'Invalid parameter for size'
        h,w = size
        mask = gen_cfa_with_black(h,w,0.2)
    else:
        raise ValueError('Unsupported cfa: {}'.format(cfa))
    if mask is None: raise ValueError('None here')
    return mask

def remosaic(ground_truth,cfa,basic_mask=None):
    if basic_mask is None:
        try:
            basic_mask = get_cfa(cfa,ground_truth.shape)
        except:
            print(ground_truth)
            exit()
    h,w,_ = ground_truth.shape
    mh,mw,_ = basic_mask.shape
    mask = basic_mask.copy()
    if mh<h:
        scale = ceil(h/mh)
        mask = np.tile(mask,(scale,1,1))
    if mw<w:
        scale = ceil(w/mw)
        mask = np.tile(mask,(1,scale,1))
    mask = mask[:h,:w]
    mosaic = mask*ground_truth
    return mosaic,mask


def get_random_patch(img,patch_size):
    h,w,_ = img.shape
    ir = 0
    ic = 0

    if h-patch_size>0:
        ir = random.randrange(0,h-patch_size+1)
        ic = random.randrange(0,h-patch_size+1)

    patch = img[ir:ir+patch_size,ic:ic+patch_size]
    return patch


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


""" #################### """
"""      condition       """
""" #################### """
def is_zero(x):
    eps = 2e-6
    return abs(x)<eps


if __name__ == "__main__":
    pass