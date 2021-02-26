import os
import cv2
import torch
import numpy as np
import torch.utils.data as data

from math import ceil
from dataset import common
from cfa_pattern import TYPE_LOC_DELTA
from torchvision.transforms import Compose, ToTensor


class LRHRDataset(data.Dataset):
    '''
    Read Mosaic and ground truth images in train and eval phases.
    '''
    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.cfa = opt['cfa']
        self.train = (opt['phase'] == 'train')
        self.paths_mosaic, self.paths_ground_truth = None, None
        self.basic_mask = None
        self.bind_patterns = dict()
        # self.bind_pattern = None

        self.no_split_cfas = [
            'RGGB',
            'Random',
            'Random16',
            'BIND41_RTN100_16',
            'RandomBlack20',
            'Random_base',
            'Random_pixel',
            'Random_2JCS',
            'Random_3JCS',
            'Random_4JCS',
            'Random_6JCS',
            'RandomFuse2',
            'RandomFuse3',
            'RandomFuse4',
            'RandomFuse6',
        ]

        self.random_type_class = {
            1:'Random_pixel',
            2:'Random_2JCS',
            3:'Random_3JCS',
            4:'Random_4JCS',
            6:'Random_6JCS',
        }

        self.capsule_cfas = [
            'Random_pixel',
            'Random_2JCS',
            'Random_3JCS',
            'Random_4JCS',
            'Random_6JCS'
        ]

        self.fusion_options = {
            'RandomFuse2':[1,2],
            'RandomFuse3':[1,2,3],
            'RandomFuse4':[1,2,3,4],
            'RandomFuse6':[1,2,4,6]
        }

        # gaussian & poisson noise combine, pixel value domain nomailized to 1
        a = opt['a']
        b = opt['b']
        self.noise_flag = not (common.is_zero(a) and common.is_zero(b))
        if self.noise_flag:
            self.sigmas= torch.tensor([torch.sqrt(i/255*a+b) for i in torch.arange(0.,256*4)])

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 1 if self.name() in ['MIT','Sandwitch'] else 2

        # read image list from image/binary files
        self.paths_ground_truth = common.get_image_paths(self.opt['datadir'])
        # self.paths_ground_truth = self.paths_ground_truth[:400]

        assert self.paths_ground_truth, '[Error] Ground truth paths are empty.'


    def name(self):
        return common.find_benchmark(self.opt['datadir'])
    

    def _reset(self):
        if self.cfa in [
            'Random',
            'Random16',
            'RandomBlack20'
        ]:
            self.basic_mask = None


    def _get_basic_mask(self,shape):
        if self.basic_mask is None:
            self.basic_mask = common.get_cfa(self.cfa,shape)
            # keep periodicity even if paving basic mask as tiles
            if self.opt['opt']=='3JCS':
                self.basic_mask = self.basic_mask[:120,:120]
        if self.basic_mask is None: 
            raise ValueError('None here')
        return self.basic_mask


    def __getitem__(self, idx):
        ground_truth, ground_truth_path = self._load_ground_truth(idx)
        # add noise for random jcs
        noise = None
        if self.noise_flag and (self.cfa in self.capsule_cfas or self.cfa in self.random_type_class):
            a,b = self.opt['a'],self.opt['b']
            if not common.is_zero(a):
                raise ValueError('parameter a is not supported for capsule cfa')
            noise = np.random.normal(loc=0,scale=np.sqrt(b),size=ground_truth.shape)
        if 'RandomFuse' in self.cfa:
            real_cfa = self.cfa
            mosaics = []
            for option in self.fusion_options[real_cfa]:
                ground_truth_ = ground_truth.copy()
                self.cfa = self.random_type_class[option]
                self.basic_mask = self._get_basic_mask(ground_truth_.shape)
                mosaic, mask = common.remosaic(ground_truth_,self.cfa,self.basic_mask)
                if noise is not None:
                    mosaic = mosaic.astype(np.float32) + noise*mask
                mosaic, ground_truth_ = self._preprocess(mosaic, ground_truth_,mask)
                mosaics.append(mosaic)
                self.basic_mask = None
            ground_truth = ground_truth_
            mosaic = torch.cat(mosaics,dim=0)
            self.cfa = real_cfa
        else:
            self.basic_mask = self._get_basic_mask(ground_truth.shape)
            mosaic, mask = common.remosaic(ground_truth,self.cfa,self.basic_mask)
            if noise is not None:
                mosaic = mosaic.astype(np.float32) + noise*mask
            mosaic, ground_truth = self._preprocess(mosaic, ground_truth,mask)
        self._reset()
        return {'mosaic': mosaic, 'ground_truth': ground_truth, 'ground_truth_path': ground_truth_path}


    def __len__(self):
        if self.train:
            return len(self.paths_ground_truth) * self.repeat
        else:
            return len(self.paths_ground_truth)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_ground_truth)
        else:
            return idx

    def _load_ground_truth(self, idx):
        idx = self._get_index(idx)
        ground_truth_path = self.paths_ground_truth[idx]
        ground_truth = common.read_img(ground_truth_path)
        opt = self.opt
        if self.train:
            patch_size = opt['patch_size']
            ground_truth = common.get_random_patch(ground_truth,patch_size=opt['patch_size'])
        if opt['augment'] is not None and opt['augment']:
            ground_truth = common.augment([ground_truth])
        return ground_truth, ground_truth_path

    
    def transform2cfa(self,img):
        """
        input: torch tensor.int16/bool in *BGR* order due to cv2.imread
        output: torch tensor.int16/int masked by cfa pattern
        """
        img = img.clone()
        cfa = self.cfa
        if cfa in self.no_split_cfas:
            pass
        elif cfa == '2JCS':
            img[:,::2,:] = img[:,::2,:]+img[:,1::2,:]
            img[:,1::2,:] = 0
        elif cfa == '3JCS':
            img[:,:,::2]=img[:,:,::2]+img[:,:,1::2]
            img[:,:-2:3,:]=img[:,:-2:3,:]+img[:,1:-2:3,:]+img[:,2:-2:3,:]
            img[:,-2,:]=img[:,-2,:]+img[:,-1,:]
            img[:,:,1::2]=0
            img[:,1::3,:]=0
            img[:,2::3,:]=0
        elif cfa == '4JCS':
            img[:,::2,:] = img[:,::2,:]+img[:,1::2,:]
            img[:,:,::2] = img[:,:,::2]+img[:,:,1::2]
            img[:,1::2,:] = 0
            img[:,:,1::2] = 0
        else:
            raise Exception("Undefined CFA \"{}\"".format(self.cfa))
        return img


    
    def transform2average(self,img,loc):
        """
        input: cfa torch tensor
        output: averaged masaicked image
        """
        img = img.clone()
        cfa = self.cfa
        if cfa in self.no_split_cfas:
            pass
        elif cfa == '2JCS':
            img[:,::2,:] = img[:,::2,:]/2
            img[:,1::2,:] = img[:,::2,:]
        elif cfa == '3JCS':
            loc = loc.type(torch.int8)
            count = self.transform2cfa(loc).type(torch.float)
            img[count>0] = img[count>0]/count[count>0]
            img[:,:,1::2] = img[:,:,::2]
            img[:,1::3,:] = img[:,::3,:]
            img[:,2::3,:] = img[:,:-2:3,:]
            img = img*loc.type(torch.float32)
        elif cfa == '4JCS':
            img = img / 4
            img[:,:,1::2] = img[:,:,::2]
            img[:,1::2,:] = img[:,::2,:]
        else:
            raise Exception("Undefined CFA:{}".format(cfa))
        img = img/255.
        img[img>1]=1.
        img[img<0]=0.
        img = img.type(torch.float32)
        return img


    def _process_capsule(self,mosaic,bind_pattern,mask,capsule_type):
        index = bind_pattern==capsule_type
        assert mosaic.dtype == np.float32
        index_list = []
        sum = np.zeros_like(mosaic).astype(np.float32)
        sum[index] = mosaic[index]

        total_idx = index.copy()
        for di,dj in TYPE_LOC_DELTA[capsule_type]:
            idx = index.copy()
            idx = np.roll(idx,shift=(di,dj,),axis=(0,1,))
            sum[index] += mosaic[idx]
            total_idx = total_idx | idx
            index_list.append(idx)

        sum = np.sum(sum,axis=2)
        sum[index] = sum[index]/(len(index_list)+1)
        
        for idx in index_list:
            sum[idx] = sum[index]
        mosaic[total_idx] = mask[total_idx]*(sum[:,:,None][total_idx])
        return mosaic


    def _process_bind16_pattern(self, mosaic, ground_truth,mask):
        bind_pattern = common.BIND_PATTERN[self.cfa]
        hb,wb = bind_pattern.shape
        mosaic = mosaic[:h//16*16,:w//16*16]
        ground_truth = ground_truth[:h//16*16,:w//16*16]
        mask = mask[:h//16*16,:w//16*16]
        h,w,c = mosaic.shape
        if hb<h:
            bind_pattern = np.tile(bind_pattern,(ceil(h/hb),1))
        if wb<w:
            bind_pattern = np.tile(bind_pattern,(1,ceil(w/wb)))
        bind = bind_pattern[:h,:w]
        self._process_capsule(mosaic,bind,mask,capsule_type=2)
        self._process_capsule(mosaic,bind,mask,capsule_type=3)

        return mosaic

    def _process_random_joint_pattern(self, mosaic, mask):
        cfa = self.cfa
        if cfa not in self.bind_patterns:
            self.bind_patterns[cfa] = np.load('cfa_pattern/{}.npy'.format(cfa))
        bind_pattern = self.bind_patterns[cfa]

        if cfa=='Random_pixel': return mosaic
        elif cfa=='Random_2JCS': capsule_types = [2,3]
        elif cfa=='Random_3JCS': capsule_types = [4,5,6,7,8,9]
        elif cfa=='Random_4JCS': capsule_types = [10,11,12,13,14]
        elif cfa=='Random_6JCS': capsule_types = [61,62,63,64,65,66,67,68,69]
        
        for capsule_type in capsule_types:
            mosaic = self._process_capsule(mosaic,bind_pattern,mask,capsule_type)
        return mosaic
    
    def _preprocess(self, mosaic, ground_truth,mask):
        if 'BIND' in self.cfa:
            mosaic = self._process_bind16_pattern(mosaic, ground_truth,mask)
        
        if self.cfa in self.random_type_class.values():
            mosaic = mosaic.astype(np.float32)
            mosaic = self._process_random_joint_pattern(mosaic,mask)

        """
        =========================↑↑↑numpy array↑↑↑=====================================
        ATTENTION, ToTensor will change dimension order and value range
        If orignal pixel value(np.uint8) range is [0,255], the converted range is [0,1]
        intput  np.int16 => tensor.int16
        intput  np.bool => tensor.bool
        target  np.uint8 => tenosr.float32   divided automatically by 255
        ========================↓↓↓torch tensor↓↓↓=====================================
        """
        transform = ToTensor()
        if self.cfa in self.random_type_class or 'BIND' in self.cfa:
            mosaic = transform(mosaic.astype(np.float32))    # np.float32 =>   tensor.float32
        else:
            mosaic = transform(mosaic.astype(np.int16))      # np.int16   =>   tensor.int16
        ground_truth = transform(ground_truth)               # np.uint8   =>   tensor.float32
        loc = transform(mask.astype(np.bool))                # np.bool    =>   tensor.bool

        # melt multiple pixels into one
        mosaic = self.transform2cfa(mosaic)

        # add noise
        if self.noise_flag and self.cfa not in self.capsule_cfas:
            mu = torch.zeros(mosaic.shape)
            noise_sigma = self.sigmas[mosaic.type(torch.long)]
            noise = (torch.normal(mu,noise_sigma)*((mosaic>0).type(torch.float32)))*255# ensure signal and noise in the same scale
            noisy_mosaic = mosaic.type(torch.float32)+noise
        else:
            noisy_mosaic = mosaic.type(torch.float32)
        
        # joint pixel split into multiple ones
        noisy_mosaic = self.transform2average(noisy_mosaic,loc)
        return noisy_mosaic, ground_truth
