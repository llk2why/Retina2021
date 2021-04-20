import os
import sys
import cv2

import numpy as np
import math
import torch
import torch.nn as nn
from network import *

from dataset import LRHRDataset
from torch.utils.data import DataLoader

def calc_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def output_JointPixel():
    device = 'cuda'
    with torch.no_grad():
        datadir = '/ssd/DemosaicDataset/patches/Kodak'

        code = 6
        opt = {
            'cfa':'RandomBaseFuse{}'.format(code),
            'a':0.0,
            'b':0.0400,
            'datadir':datadir,
            'patch_size':128,
            'augment':None,
            'phase':'train',
        }

        test_dataset = LRHRDataset(opt)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=10, 
            pin_memory=True
        )

        
        template_degree = {
            2:2,
            3:3,
            4:4,
            6:4
        }
        fusion_degree = template_degree[code]
        net = JointPixel_fusion(debug=True,fusion_degree=fusion_degree).to(device)
        net.eval()
        ckpt = 'checkpoints/RandomBaseFuse{}/001_JointPixel_MIT_a=0.0000_b=0.0400/epochs/best_ckp.pth'.format(code)
        checkpoint = torch.load(ckpt)['state_dict']
        from collections import OrderedDict
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            namekey = k[7:]
            new_checkpoint[namekey] = v
        net.load_state_dict(new_checkpoint)
        

        for batch in test_loader:
            print(batch.keys())
            x,y = batch['mosaic'],batch['ground_truth']
            x,y = x.to(device),y.to(device)
            fusion_stage3_right_output,print_fusion_stage3_right_output,fuse_mid,stage5_input,stage5_output = net(x)
            print(len(fusion_stage3_right_output),fusion_stage3_right_output[0].shape)
            print(fuse_mid.shape)
            print(stage5_input.shape)
            print(stage5_output.shape)
            dc = {
                # 'mid1':fusion_stage3_right_output[0],
                # 'mid2':fusion_stage3_right_output[1],
                'fusemid':fuse_mid,
                'result':stage5_output
            }
            for i in range(len(fusion_stage3_right_output)):
                dc['mid{}'.format(i+1)] = fusion_stage3_right_output[i]
            for i in range(len(print_fusion_stage3_right_output)):
                dc['print{}'.format(i+1)] = print_fusion_stage3_right_output[i]
            
            os.makedirs('mid_results/{}'.format(code),exist_ok=True)
            with open('mid_results/{}.txt'.format(code),'w') as f:
                for k in dc:
                    print('max:',dc[k][0].max())
                    dc[k] = dc[k][0].cpu().numpy().transpose(1,2,0)
                    dc[k] = np.abs(dc[k])
                    img = (dc[k] * 255)
                    img[img > 255] = 255
                    img[img < 0] = 0
                    img = img.astype(np.uint8)
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    cv2.imwrite('mid_results/{}/{}.png'.format(code,k),img)
                    f.write('{}:{}\n'.format(k,img.max()))
            break


def start():
    output_JointPixel()
        
    

if __name__ == "__main__":
    start()