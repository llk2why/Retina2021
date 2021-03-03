import os
import cv2
import sys
import torch
import numpy as np

from torchvision.transforms import Compose, ToTensor

file_path = os.path.abspath(__file__)
dst_dir = os.path.abspath(__file__+'/../..')
os.chdir(dst_dir)
print('changing workspace to %s ...' % os.getcwd())
sys.path.append(os.getcwd())

from dataset.LRHR_dataset import LRHRDataset
from cfa_pattern import TYPE_LOC_DELTA

opt = {
    'cfa':'2JCS',
    'phase':'train',
    'a':0.0,
    'b':0.0,
    'datadir':'/ssd/DemosaicDataset/test_small_dataset',
    'patch_size':128,
    'augment':False
}

random_type_class = {
    1:'Random_pixel',
    2:'Random_2JCS',
    3:'Random_3JCS',
    4:'Random_4JCS',
    6:'Random_6JCS',
}

capsule_cfas = [
    'Random_pixel',
    'Random_2JCS',
    'Random_3JCS',
    'Random_4JCS',
    'Random_6JCS'
]

fusion_options = {
    'RandomFuse2':[1,2],
    'RandomFuse3':[1,2,3],
    'RandomFuse4':[1,2,3,4],
    'RandomFuse6':[1,2,4,6]
}


def check_mosaic_groundtruth_match(mosaic,ground_truth):
    plain_mosaic = mosaic[:3].clone()
    index = plain_mosaic>0
    res = (plain_mosaic[index]==ground_truth[index]).all()
    print(res)
    if not res: print('mosaic groundtruth not match!!!')
    return res.item()


def is_strict_equal(a,b):
    return np.abs(a-b)<1e-6


def is_equal(a,b):
    return np.abs(a-b)<=1e-4


def check_capsule_type(mosaic,bind_pattern,ground_truth,mask,capsule_type,ground_truth_path):
        index = torch.from_numpy(bind_pattern==capsule_type)
        h,w = index.shape

        total_idx = index.clone()
        for di,dj in TYPE_LOC_DELTA[capsule_type]:
            idx = index.clone()
            idx = torch.roll(idx,shifts=[di,dj],dims=[0,1])
            total_idx |= idx

        sum2 = torch.zeros(h,w)
        sum2[total_idx] = ((mask*ground_truth).sum(dim=0))[total_idx]

        sum1 = torch.zeros(h,w)
        # sum1 = mosaic.sum(dim=0)
        # print('\ncapsule type:',capsule_type)
        # print(mosaic.shape)
        # tmp = torch.sum(mosaic,dim=0)
        # print(tmp[0:2,8:12])
        # print(tmp.shape)
        sum1[total_idx] = (torch.sum(mosaic,dim=0))[total_idx]
        
        for i in range(h):
            for j in range(w):
                if index[i,j]:
                    # print(capsule_type,index[i,j])
                    s1 = 0+sum1[i,j].item()
                    s2 = 0+sum2[i,j].item()

                    x = max(0,i-2)
                    y = min(h-1,i+3)
                    u = max(0,j-2)
                    v = min(w-1,j+3)
                    sa = slice(x,y)
                    sb = slice(u,v)

                    for di,dj in TYPE_LOC_DELTA[capsule_type]:
                        ii,jj = i+di,j+dj
                        # assert is_strict_equal(sum1[i,j],sum1[ii,jj]),\
                        #     '({},{}):{} not equals to ({},{}):{}'.format(i,j,sum1[i,j],ii,jj,sum1[ii,jj])
                        if not  is_strict_equal(sum1[i,j],sum1[ii,jj]):
                            for p in range(x,y):
                                for q in range(u,v):
                                    print("({},{})".format(str(p).zfill(3),str(q).zfill(3)),end=" ")
                                print('')
                            # print('not self equals')
                            print(sum1.shape)
                            print(total_idx[sa,sb])
                            print('【not self equals】:({},{}):{} not equals to ({},{}):{}'.format(i,j,sum1[i,j],ii,jj,sum1[ii,jj]))
                            print('bind_pattern:')
                            print(bind_pattern[sa,sb])
                            print('mosaic:')
                            print(mosaic[0,sa,sb])
                            print(mosaic[1,sa,sb])
                            print(mosaic[2,sa,sb])

                            print('mosaic sum:')
                            print(sum1[sa,sb])
                            print('ground truth:')
                            print((ground_truth*mask)[0,sa,sb])
                            print((ground_truth*mask)[1,sa,sb])
                            print((ground_truth*mask)[2,sa,sb])
                            print('\n\n')
                            return False
                            exit()
                        
                        s1 += sum1[ii,jj].item()
                        s2 += sum2[ii,jj].item()
                    torch.set_printoptions(precision=9)
                    if not  is_equal(s1,s2):
                        for p in range(x,y):
                            for q in range(u,v):
                                print("({},{})".format(str(p).zfill(3),str(q).zfill(3)),end=" ")
                            print('')
                        print('【not mosaic ground truth】({},{}):{:.6f} not equals to ({},{}):{:.6f}  delta:{:.4f}'.format(i,j,s1,ii,jj,s2,np.abs(s1-s2)*255))
                        sa = slice(x,y)
                        sb = slice(u,v)
                        print('bind_pattern:')
                        print(bind_pattern[sa,sb])
                        
                        print('mosaic:')
                        print(mosaic[0,sa,sb])
                        print(mosaic[1,sa,sb])
                        print(mosaic[2,sa,sb])

                        print('ground truth:')

                        
                        print((ground_truth*mask)[0,sa,sb])
                        print((ground_truth*mask)[1,sa,sb])
                        print((ground_truth*mask)[2,sa,sb])
                        # print(mosaic.dtype)
                        # print(ground_truth.dtype)

                        im = cv2.imread(ground_truth_path)
                        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                        im_tensor = ToTensor()(im)
                        masked_tensor = im_tensor*mask
                        im_tensor_masked = torch.zeros(h,w).type(torch.float32)
                        im_tensor_masked[total_idx] = masked_tensor.sum(dim=0)[total_idx]
                        s = im_tensor_masked[i,j].item()
                        for di,dj in TYPE_LOC_DELTA[capsule_type]:
                            ii,jj = i+di,j+dj
                            s += im_tensor_masked[ii,jj]
                        # print('mosaic sum:')
                        # print(sum1[sa,sb])
                        # print('ground truth sum2:')
                        # print(sum2[sa,sb])
                        # print('sum3:')
                        # print(im_tensor_masked[sa,sb])
                        # print('理应的数值(tensor):{:.6f}'.format(s))
                        # print('new load truth:')
                        # print(masked_tensor[0,sa,sb])
                        # print(masked_tensor[1,sa,sb])
                        # print(masked_tensor[2,sa,sb])

                        # im_numpy = im.astype(np.float32)/255
                        # im_numpy = ToTensor()(im)
                        # masked_tensor = im_numpy*mask
                        # im_tensor_masked = torch.zeros(h,w).type(torch.float32)
                        # im_tensor_masked[total_idx] = masked_tensor.sum(dim=0)[total_idx]
                        # s = im_tensor_masked[i,j].item()
                        # for di,dj in TYPE_LOC_DELTA[capsule_type]:
                        #     ii,jj = i+di,j+dj
                        #     s += im_tensor_masked[ii,jj]
                        # print('mosaic sum:')
                        # print(sum1[sa,sb])
                        # print('ground truth sum2:')
                        # print(sum2[sa,sb])
                        # print('sum3:')
                        # print(im_tensor_masked[sa,sb])
                        # print('理应的数值(tensor):{:.6f}'.format(s))
                        # print('new load truth:')
                        # print(masked_tensor[0,sa,sb])
                        # print(masked_tensor[1,sa,sb])
                        # print(masked_tensor[2,sa,sb])
                        # print(sa,sb)
                        print('')
                        print('\n\n')
                        return False
                        exit()
                        
        return True
                        


def check_capsule_class(cfa,jcs,ground_truth,ground_truth_path):
    print(cfa)
    if cfa=='Random_pixel': True
    elif cfa=='Random_2JCS': capsule_types = [2,3]
    elif cfa=='Random_3JCS': capsule_types = [4,5,6,7,8,9]
    elif cfa=='Random_4JCS': capsule_types = [10,11,12,13,14]
    elif cfa=='Random_6JCS': capsule_types = [61,62,63,64,65,66,67,68,69]
    else: raise ValueError('unsupported cfa')
    bind_pattern = np.load('cfa_pattern/{}.npy'.format(cfa))
    basic_mask = np.load('cfa_pattern/random_base128.npy')
    basic_mask = cv2.cvtColor(basic_mask,cv2.COLOR_BGR2RGB)
    mask = (basic_mask*((bind_pattern!=0)[:,:,None])).astype(np.bool)
    mask = ToTensor()(mask)
    c,h,w = mask.shape
    
    # flags = ((mask==0) == (jcs==0))
    invert_mask = ~(mask>0)
    flag = ((invert_mask*jcs)>0).all()
    
    if flag:
        print('%s pattern doesn\'t match mask' % cfa)
        return False
    for capsule_type in capsule_types:
        mid_res = check_capsule_type(jcs,bind_pattern,ground_truth,mask,capsule_type,ground_truth_path)
        if not mid_res:return False
    return True


def check_joint_cfa(mosaic,ground_truth,ground_truth_path):
    res = True
    cfa = opt['cfa']
    if cfa in capsule_cfas:
        res = check_capsule_type(cfa,mosaic,ground_truth)
    elif cfa in fusion_options:
        for i,capsule_class in enumerate(fusion_options[cfa]):
            if capsule_class==1: continue
            jcs = mosaic[i*3:i*3+3].clone()
            mid_cfa = random_type_class[capsule_class]
            mid_res = check_capsule_class(mid_cfa,jcs,ground_truth,ground_truth_path)
            res &= mid_res
            if not res: break
    return res


def check_ground_truth(ground_truth,ground_truth_path):
    im = cv2.imread(ground_truth_path)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im_tensor = ToTensor()(im)
    return (ground_truth==im_tensor).all().item()


def check_fix_joint(mosaic,ground_truth,ground_truth_path):
    # print(ground_truth.shape)
    # im_mosaic = mosaic.numpy().transpose(1,2,0)*255
    # im_mosaic = im_mosaic.astype(np.uint8)
    # im_mosaic = cv2.cvtColor(im_mosaic,cv2.COLOR_RGB2BGR)
    # cv2.imwrite('{}.png'.format(opt['cfa']),im_mosaic)
    sum = mosaic.sum(dim=0)
    h,w = sum.shape
    mask = mosaic>0
    masked_ground_truth = ground_truth*mask
    sum2 = masked_ground_truth.sum(dim=0)
    if opt['cfa'] == '2JCS':
        sum2[::2] = (sum2[0::2]+sum2[1::2])/2
        sum2[1::2] = sum2[::2]
    for i in range(h):
        for j in range(w):
            if opt['cfa'] == '2JCS':
                if i%2==0:
                    assert sum[i,j] == sum[i+1,j]
                assert torch.abs(sum[i,j]-sum2[i,j])<1e-4
            elif opt['cfa'] == '3JCS':
                if i%3==0:
                    if j%4==0:
                        if j+1 <w: assert sum[i,j] == sum[i,j+1]
                        if i+1 < h and j+1<w: assert sum[i,j] == sum[i+1,j+1]
                    if j%4==2:
                        if j+1<w: assert sum[i,j] == sum[i,j+1]
                        if i+1<h: assert sum[i,j] == sum[i+1,j]
                if i%3==1:
                    if j%4==0:
                        if i+1 < h: assert sum[i,j] == sum[i+1,j]
                        if i+1 < h and j+1<w: assert sum[i,j] == sum[i+1,j+1]
                    if j%4==3:
                        if i+1 < h and j-1>0: assert sum[i,j] == sum[i+1,j-1]
                        if i+1<h: assert sum[i,j] == sum[i+1,j]
            elif opt['cfa'] == '4JCS':
                if i%2==0 and j%2==0:
                    assert sum[i,j] == sum[i+1,j] and \
                           sum[i,j] == sum[i,j+1] and \
                           sum[i,j] == sum[i+1,j+1]
    return True
        


def verify():
    dataset = LRHRDataset(opt)
    res = True
    for i in range(dataset.__len__()):
        items = dataset.__getitem__(i)
        mosaic = items['mosaic']
        ground_truth = items['ground_truth']
        ground_truth_path = items['ground_truth_path']
        print(ground_truth_path)
        res1 = check_ground_truth(ground_truth,ground_truth_path)
        if opt['cfa'] not in ['2JCS','3JCS','4JCS']:
            res2 = check_mosaic_groundtruth_match(mosaic,ground_truth)
        else:
            res2 = True
        if opt['cfa'] in capsule_cfas:
            res3 = check_joint_cfa(mosaic,ground_truth,ground_truth_path)
        elif opt['cfa'] in ['2JCS','3JCS','4JCS']:
            res3 = check_fix_joint(mosaic,ground_truth,ground_truth_path)
        else:
            raise ValueError('unsupported cfa')
        res &= res1
        res &= res2
        res &= res3
        if not res: break
    return res



def main():
    res = verify()
    if res: print('dataset test PASSED!')
    else: print('dataset test Failed!')


if __name__ == '__main__':
    main()