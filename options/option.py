import os
import json
import torch
from datetime import datetime
from collections import OrderedDict

from utils import get_timestamp,mkdir_and_rename,mkdirs

def parse(args,rank=None,world_size=None):    
    # remove comments starting with '//'
    json_str = ''
    with open(args.opt,'r') as f:
        for line in f:
            line = line.strip().split('//')[0]+'\n'
            json_str += line
    opt = json.loads(json_str,object_pairs_hook=OrderedDict)
    opt['is_train'] = args.is_train
    opt['save_image'] = args.save_image
    opt['network'] = args.network
    opt['pretrained_path'] = args.pretrained_path
    opt['cfa'] = args.cfa
    opt['a'] = args.a
    opt['b'] = args.b
    opt['rank'] = rank
    opt['world_size'] = world_size

    opt['timestamp'] = get_timestamp()
    
    if not torch.cuda.is_available():
        raise ValueError('Only GPU mode is supported')
    
    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
            
    
    # id is the experiment id to distinguish different experiments with different hyper-parameters.
    config_str = '{}/{}_{}_{}_a={:.4f}_b={:.4f}'.format(
        opt['cfa'],
        str(opt['id']).zfill(3),
        opt['network'],
        opt['dataset_name'],
        opt['a'],
        opt['b']
    )
    ckpt_path = os.path.join(os.getcwd(), 'checkpoints', config_str)

    ckpt_path = os.path.relpath(ckpt_path)

    path_opt = OrderedDict()
    path_opt['ckpt_root'] = ckpt_path
    path_opt['epochs'] = os.path.join(ckpt_path, 'epochs')
    path_opt['visual'] = os.path.join(ckpt_path, 'visual')
    path_opt['records'] = os.path.join(ckpt_path, 'records')
    opt['path'] = path_opt
    
    if args.is_train:
        if rank==0:
            # create folders
            archived_path = mkdir_and_rename(opt['path']['ckpt_root'])  # rename old experiments if exists
            mkdirs((path for key, path in opt['path'].items() if not key == 'ckpt_root'))

            # if different, there's an old checkpoint folder
            # copy for a smoother resume process, in case of "pretrained_path NOT FOUND" 
            if opt['path']['ckpt_root'] != archived_path and args.pretrained_path is not None:
                dst = args.pretrained_path
                src = os.path.join(archived_path,'epochs',os.path.basename(dst))
                print(src)
                print(dst)
                import shutil; shutil.copy2(src,dst)

            save(opt)

        print("===> Experimental DIR: [%s]"%ckpt_path)

    opt = dict_to_nonedict(opt)
    return opt


def save(opt):
    dump_dir = opt['path']['ckpt_root']
    dump_path = os.path.join(dump_dir, 'options.json')
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=4)
    

class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt