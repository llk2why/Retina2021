import os
import cv2
import tqdm
import json

import multiprocessing

def save_img(fpath,im):
    cv2.imwrite(fpath,im)

def gen_patch(img,basename,dst_dir,patch_size=128,stride=64):
    r, c, _ = img.shape
    cnt = 0
    patches = []
    irange = list(range(0, r - patch_size, stride))
    jrange = list(range(0, c - patch_size, stride))
    if irange[-1] + patch_size != r:
        irange.append(r - patch_size)
    if jrange[-1] + patch_size != c:
        jrange.append(c - patch_size)
    processes = 4
    pool = multiprocessing.Pool(processes)
    for i in irange:
        for j in jrange:
            name,suffix = os.path.splitext(basename)
            fpath = os.path.join(dst_dir,'{}_{}.png'.format(name,str(cnt).zfill(4)))
            im = img[i:i + patch_size, j:j + patch_size]
            if not os.path.exists(fpath):
                pool.apply_async(save_img,(fpath,im,))
            patches.append([[i, j], [i + patch_size, j + patch_size]])
            cnt += 1
            assert im.shape == (patch_size, patch_size, 3), '{}  {}'.format(im.shape, patch_size)
    pool.close()
    pool.join()
    return patches


def chop_dir(src_dir,src_prefix,dst_prefix):
    dst_dir = src_dir.replace(src_prefix,dst_prefix)
    os.makedirs(dst_dir,exist_ok=True)
    patch_info = dict()
    for png_name in tqdm.tqdm(os.listdir(src_dir)):
        png_path = os.path.join(src_dir,png_name)
        basename = os.path.basename(png_path)
        img = cv2.imread(png_path)
        patches = gen_patch(img,basename,dst_dir)

        patch_info[basename] = dict()
        patch_info[basename]['patches'] = patches
        patch_info[basename]['size'] = list(img.shape)
    
    json.dump(patch_info,open(dst_dir+'/patch_info.json','w'))


def main():
    src_prefix = '/ssd/DemosaicDataset'
    dst_prefix = '/ssd/DemosaicDataset/patches'
    src_dirs = [
        '/ssd/DemosaicDataset/Kodak',
        '/ssd/DemosaicDataset/McM',
    ]
    for src_dir in src_dirs:
        chop_dir(src_dir,src_prefix,dst_prefix)

if __name__ == '__main__':
    main()