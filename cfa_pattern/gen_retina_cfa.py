import os
import sys
import cv2
import numpy as np

from config import TYPE_LOC_DELTA

pattern_list = [
    'Random_pixel',
    'Random_2JCS',
    'Random_3JCS',
    'Random_4JCS',
    'Random_6JCS'
]

def gen_random_base128():
    if os.path.exists('random_base128.npy'):
        return
    num=128*128//4
    pattern = np.array([0]*num+[1]*(num*2)+[2]*num)
    np.random.shuffle(pattern)
    np.random.shuffle(pattern)
    pattern = pattern.reshape(128,128)
    ones = np.ones((128,128,3))
    mask = np.stack([np.where(pattern == i, ones[:, :, i], 0) for i in range(3)], axis=-1)
    np.save('random_base128.npy',mask.astype(np.uint8))
    print(mask.max())
    cv2.imwrite('images/plain_pattern/random_128.png',mask*255)


def check_valid(pattern,capsule_type,i,j):
    if capsule_type==1: return True
    for di,dj in TYPE_LOC_DELTA[capsule_type]:
        ii,jj = i+di,j+dj
        if ii<0 or ii>=128 or jj<0 or jj>=128:
            return False
        if pattern[ii,jj]!=0:
            return False
    return True


def fill_capsule(pattern,capsule_type,i,j):
    pattern[i][j]=capsule_type
    for di,dj in TYPE_LOC_DELTA[capsule_type]:
        ii,jj = i+di,j+dj
        pattern[ii][jj]=-capsule_type

def get_cover():
    cover = np.zeros((128,128)).astype(np.bool)
    for pattern_name in pattern_list:
        npy_path = pattern_name+'.npy'
        if os.path.exists(npy_path):
            pattern = np.load(npy_path)
            cover = cover | (pattern!=0)
    return cover

def _gen_random_pattern(capsule_class,ratio,retain=True):
    p = ratio
    if capsule_class == 1:
        candidates = [1]
        name = 'Random_pixel'
    elif capsule_class == 2:
        candidates = [2,3]
        name = 'Random_2JCS'
    elif capsule_class == 3:
        candidates = [4,5,6,7,8,9]
        name = 'Random_3JCS'
    elif capsule_class == 4:
        candidates = [10,11,12,13,14]
        name = 'Random_4JCS'
    elif capsule_class == 6:
        candidates = [61,62,63,64,65,66,67,68,69]
        name = 'Random_6JCS'
    else:
        raise ValueError('unsupported joint number:{}'.format(capsule_class))
    npy_name = name+'.npy'
    png_name = name+'.png'
    png_path = 'images/plain_pattern/'+png_name

    if retain and os.path.exists(npy_name): return

    joint_pattern = np.zeros((128,128))
    random_mat = np.random.uniform(0,1,size=(128,128))
    cover = get_cover()

    # directly sample subsamping pixel.
    if name != 'Random_pixel':
        for i in range(128):
            for j in range(128):
                if cover[i][j]==True or joint_pattern[i,j]!=0: continue
                capsule_type = np.random.choice(candidates)
                if not check_valid(joint_pattern,capsule_type,i,j):
                    if name == 'Random_2JCS':
                        capsule_type = 5-capsule_type
                        if not check_valid(joint_pattern,capsule_type,i,j):
                            continue
                    else:
                        continue
                fill_capsule(joint_pattern,capsule_type,i,j)

    for i in range(128):
        for j in range(128):
            # sampling from a probability distribution
            if random_mat[i][j]>p or joint_pattern[i,j]!=0: continue
            capsule_type = np.random.choice(candidates)
            if not check_valid(joint_pattern,capsule_type,i,j):continue
            fill_capsule(joint_pattern,capsule_type,i,j)
    
    joint_pattern = joint_pattern.astype(np.int8)
    np.save(npy_name,joint_pattern)

    base_pattern = np.load('random_base128.npy')
    mask = (joint_pattern!=0)[:,:,None]*base_pattern
    cv2.imwrite(png_path,mask*255)

    print('{} ratio:'.format(name),np.sum(joint_pattern!=0)/(128*128))


def gen_random_pixel(ratio=0.8,retain=True):
    _gen_random_pattern(1,ratio=ratio,retain=retain)


def gen_random_2JCS(ratio=0.5,retain=True):
    _gen_random_pattern(2,ratio=ratio,retain=retain)


def gen_random_3JCS(ratio=0.5,retain=True):
    _gen_random_pattern(3,ratio=ratio,retain=retain)


def gen_random_4JCS(ratio=0.5,retain=True):
    _gen_random_pattern(4,ratio=ratio,retain=retain)

def gen_random_6JCS(ratio=0.5,retain=True):
    _gen_random_pattern(6,ratio=ratio,retain=retain)


def check_conflict():
    for pattern_name in pattern_list:
        npy_path = pattern_name+'.npy'
        if not os.path.exists(npy_path): continue
        pattern = np.load(npy_path)
        if '6' in pattern_name:
            print(pattern[10:28,0:10])
        conflict_flag = False
        for i in range(128):
            for j in range(128):
                capsule_type = pattern[i,j]
                if capsule_type>0:
                    for di,dj in TYPE_LOC_DELTA[capsule_type]:
                        ii,jj = i+di,j+dj
                        if ii<0 or ii>=128 or jj<0 or jj>=128 or \
                           pattern[ii,jj]!=-capsule_type:
                            conflict_flag=True

                if conflict_flag: break
            if conflict_flag: break
        print(pattern_name,end=': ')
        if conflict_flag: print('The test failed')
        else: print('The test passed')

def count_capsule_type():
    for pattern_name in pattern_list:
        npy_path = pattern_name+'.npy'
        if not os.path.exists(npy_path): continue
        pattern = np.load(npy_path)
        unique, counts = np.unique(pattern, return_counts=True)
        counter = dict(zip(unique, counts))
        print(pattern_name)
        print(counter)
        print('ratio:','{:.2f}%'.format(np.sum(pattern!=0)/(128*128)*100))
        print('\n')


def check_coverage():
    mask = get_cover()
    coverage = float(mask.sum()/np.prod(mask.shape))
    print('coverage: {:.2f}%'.format(coverage*100))

    cover_png_name = 'images/plain_pattern/cover.png'
    uncover_png_name = 'images/plain_pattern/uncover.png'
    base_pattern = np.load('random_base128.npy')
    cover = (mask!=0)[:,:,None]*base_pattern
    uncover = (mask==0)[:,:,None]*base_pattern
    cover = np.repeat(cover,10,axis=0)
    cover = np.repeat(cover,10,axis=1)
    uncover = np.repeat(uncover,10,axis=0)
    uncover = np.repeat(uncover,10,axis=1)
    cv2.imwrite(cover_png_name,cover*255)
    cv2.imwrite(uncover_png_name,uncover*255)
    return coverage


def remove_npy():
    os.system('rm Random_pixel.npy')
    os.system('rm 2JCS_pixel.npy')
    os.system('rm 3JCS_pixel.npy')
    os.system('rm 4JCS_pixel.npy')
    os.system('rm 6JCS_pixel.npy')


def main():
    while True:
        gen_random_base128()
        # remove_npy()
        retain_list = [True,True,True,True,True,True]
        npy_list = [
            'Random_pixel.npy',
            'Random_2JCS.npy',
            'Random_3JCS.npy',
            'Random_4JCS.npy',
            'Random_6JCS.npy'
        ]
        for flag,npy_path in zip(retain_list,npy_list):
            if not flag and os.path.exists(npy_path):
                os.remove(npy_path)
        # gen_random_pixel(ratio=0.8,retain=retain_list[0])
        # gen_random_2JCS(ratio=0.85,retain=retain_list[1])
        # gen_random_3JCS(ratio=0.99,retain=retain_list[2])
        # gen_random_4JCS(ratio=0.99,retain=retain_list[3])
        gen_random_6JCS(ratio=0.99,retain=retain_list[4])
        check_conflict()
        count_capsule_type()
        coverage = check_coverage()
        print(coverage,type(coverage),coverage >= 0.9)
        if coverage>0.99: break
        print('\n\n\n')

if __name__ == '__main__':
    main()