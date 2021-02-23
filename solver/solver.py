import os
import cv2
import pandas as pd
import scipy.misc as misc

import torch
import torch.nn as nn
import torchvision.utils
import torch.optim as optim
import torch.distributed as dist

from utils import util
from collections import OrderedDict
from torch.cuda.amp import autocast,GradScaler

from .common import init_weights
from .common import create_model
from .base_solver import BaseSolver
from torch.cuda.amp import autocast,GradScaler


def reduce_loss(loss):
    ret = loss.clone()
    dist.all_reduce(ret,op=dist.ReduceOp.SUM)
    ret /= dist.get_world_size()
    return ret


class Solver(BaseSolver):
    def __init__(self, opt):
        super(Solver, self).__init__(opt)
        self.train_opt = opt['solver']
        self.mosaic = self.Tensor()
        self.ground_truth = self.Tensor()
        self.demosaicked = None
        self.dataset = None
        self.test_pad = None

        self.records = {
            'train_loss': [],
            'val_loss': [],
            'psnr': [],
            'ssim': [],
            'lr': []
        }

        self.model = create_model(opt)

        if self.is_train:
            self.model = self.model.cuda(opt['rank'])
            self.model = nn.parallel.DistributedDataParallel(self.model,device_ids=[opt['rank']])
            self.model.train()

            # set loss
            loss_type = self.train_opt['loss_type']
            if loss_type == 'l1':
                self.criterion_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion_pix = nn.MSELoss()
            else:
                raise NotImplementedError('Loss type [{}] is not implemented!'.format(loss_type))

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer = optim.Adam(self.model.parameters(),
                                            lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            else:
                raise NotImplementedError('Loss type [{}] is not implemented!'.format(optim_type))

            # set lr_scheduler
            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('Only MultiStepLR scheme is supported!')

            # set a GradScaler
            self.scaler = GradScaler()
        else:
            self.model = self.model.cuda()
            # self.test_pad = 24
            # TODO: check whether needs padding operation
            self.pad_ = nn.ZeroPad2d(self.test_pad)


        self.load()
        self.print_network()

        print('===> Solver Initialized : [{}] || Use GPU : [{}]'.format(self.__class__.__name__, self.use_gpu))
        if self.is_train:
            print("optimizer: ", self.optimizer)
            print("lr_scheduler milestones: {}   gamma: {}".format(self.scheduler.milestones, self.scheduler.gamma))


    def _net_init(self, init_type='kaiming'):
        print('==> Initializing the network using [{}]'.format(init_type))
        init_weights(self.model, init_type)


    def register_dataset(dataset):
        self.dataset = dataset


    def unregister_dataset(dataset):
        self.dataset = None


    def feed_data(self, batch, need_HR=True):
        input = batch['mosaic']
        self.mosaic.resize_(input.size()).copy_(input)

        if need_HR:
            target = batch['ground_truth']
            self.ground_truth.resize_(target.size()).copy_(target)


    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        loss_batch = 0.0

        with autocast():
            if self.opt['network']=='JointPixelMax':
                mid_output,noise_output = self.model(self.mosaic)
                loss_demosaick = self.criterion_pix(mid_output,self.ground_truth)
                loss_detail = self.criterion_pix(noise_output,self.ground_truth-mid_output)
                
                # TODO: check which is correct
                # loss_batch = (loss_demosaick.item()+loss_detail.item())
                loss_batch = loss_demosaick+loss_detail

            else:
                # print(self.mosaic.max())
                # print(self.ground_truth.max())
                # exit()
                output = self.model(self.mosaic)
                loss_batch = self.criterion_pix(output, self.ground_truth)

        loss_batch = reduce_loss(loss_batch)
        display_loss_batch = loss_batch.item()
        self.scaler.scale(loss_batch).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.model.eval()
        return display_loss_batch


    def test(self):
        self.model.eval()
        with torch.no_grad():
            forward_func = self.model.forward
            if self.test_pad is not None:
                n,c,h,w = self.mosaic.shape
                self.mosaic = self.pad_(self.mosaic)
            
            if self.self_ensemble and not self.is_train:
                # TODO: modify mosaic preprocessing for self_ensemble operation
                # 只需要在dataset函数里面添加一个接口函数即可，输入是groundtruth，返回
                assert self.dataset is not None
                SR = self._forward_x8(self.ground_truth, forward_func)
            else:
                SR = forward_func(self.mosaic)

            if self.test_pad is not None:
                pad = self.test_pad
                SR = SR[:,:,pad:pad+h,pad:pad+w]

            if self.opt['network']=='JointPixelMax':
                mid_output,noise_output = SR
                SR = mid_output+noise_output

            if isinstance(SR, list):
                self.demosaicked = SR[-1]
            else:
                self.demosaicked = SR
            
            # print(self.model.__class__.__name__)
            # print(self.mosaic.shape)
            # print(self.demosaicked.shape)
            # exit()

            # mosaic_output = self.mosaic.float().cpu().squeeze()[:,:,:]
            # ground_truth_output = self.ground_truth.float().cpu().squeeze()
            # demosaicked_output = self.demosaicked.float().cpu().squeeze()

            # import numpy as np
            # if os.path.exists('tmp.npy'):
            #     print('mosaic_output:')
            #     tmp_np = np.load('tmp.npy')
            #     flag = np.all(np.array_equal(tmp_np,mosaic_output))
            #     if flag: print('equal!!')
            #     else: print('unequal!!!!')
            # else:
            #     np.save('tmp.npy',mosaic_output)

            # if os.path.exists('tmp2.npy'):
            #     print('demosaicked_output:')
            #     tmp_np = np.load('tmp2.npy')
            #     flag = np.all(np.array_equal(tmp_np,demosaicked_output))
            #     if flag: print('equal!!')
            #     else: print('unequal!!!!')
            # else:
            #     np.save('tmp2.npy',demosaicked_output)

            # if os.path.exists('tmp3.npy'):
            #     print('ground_truth_output:')
            #     tmp_np = np.load('tmp3.npy')
            #     flag = np.all(np.array_equal(tmp_np,ground_truth_output))
            #     if flag: print('equal!!')
            #     else: print('unequal!!!!')
            # else:
            #     np.save('tmp3.npy',ground_truth_output)

            # print(mosaic_output.shape)
            # print(ground_truth_output.shape)
            # print(demosaicked_output.shape)

            # mosaic_output,ground_truth_output,demosaicked_output = \
            #     util.tensorToNumpy([mosaic_output,ground_truth_output,demosaicked_output])

            # print(mosaic_output.shape)
            # print(ground_truth_output.shape)
            # print(demosaicked_output.shape)

            # # mosaic_output = cv2.cvtColor(mosaic_output,cv2.COLOR_RGB2BGR)
            # ground_truth_output = cv2.cvtColor(ground_truth_output,cv2.COLOR_RGB2BGR)
            # demosaicked_output = cv2.cvtColor(demosaicked_output,cv2.COLOR_RGB2BGR)

            # # cv2.imwrite('mosaic.png',mosaic_output)
            # cv2.imwrite('ground_truth.png',ground_truth_output)
            # cv2.imwrite('demosaicked.png',demosaicked_output)
            # exit()
            

        # self.model.train()
        if self.is_train:
            loss_pix = self.criterion_pix(self.demosaicked, self.ground_truth)
            return loss_pix.item()

    # TODO:
    # collaborate with interface function in dataset instance
    def _forward_x8(self, x, forward_function):
        """
        self ensemble
        """
        def _transform(v, op):
            ret = v.clone()

            if op == 'v':
                ret = ret.flip(dims=[2])
            elif op == 'h':
                ret = ret.flip(dims=[3])
            elif op == 't':
                ret = ret.permute(0,1,3,2)

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        
        lr_list = [util.bayer(x) for x in lr_list]
    
        sr_list = []
        for aug in lr_list:
            sr = forward_function(aug)
            if isinstance(sr, list):
                sr_list.append(sr[-1])
            else:
                sr_list.append(sr)

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter('runs/images1')
        # for i,img in enumerate(lr_list):
        #     writer.add_images('imagex8_mosaic/{}'.format(i), img, 0)
        # writer.add_images('imagex8_demosaicked', output_cat, 0)
        # print(output_cat.shape)
        # writer.flush()
        # writer.close()
        # exit()
        output = output_cat.mean(dim=0, keepdim=True)

        return output


    def save_checkpoint(self, epoch, is_best):
        """
        save checkpoint to experimental dir
        """
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        print('===> Saving last checkpoint to [%s] ...]'%filename)
        ckp = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
            'records': self.records
        }

        if dist.get_rank()==0:
            torch.save(ckp, filename)
            if is_best:
                print('===> Saving best checkpoint to [%s] ...]' % filename.replace('last_ckp','best_ckp'))
                torch.save(ckp, filename.replace('last_ckp','best_ckp'))

            if epoch % self.train_opt['save_ckp_step'] == 0:
                print('===> Saving checkpoint [{}] to [{}] ...]'.format(epoch,
                                                                    filename.replace('last_ckp','epoch_%d_ckp'%epoch)))

                torch.save(ckp, filename.replace('last_ckp','epoch_%d_ckp'%epoch))


    def load(self):
        """
        load or initialize network
        """
        if (self.is_train and self.opt['pretrained_path'] is not None) or not self.is_train:
            model_path = self.opt['pretrained_path']
            if model_path is None: 
                raise ValueError("[Error] The 'pretrained_path' hasn't been provided as a parameter")

            print('===> Loading model from [{}]...'.format(model_path))
            if self.is_train:
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['state_dict'])

                self.cur_epoch = checkpoint['epoch'] + 1
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']
                self.best_epoch = checkpoint['best_epoch']
                self.records = checkpoint['records']

            else:
                device = torch.device("cuda")
                checkpoint = torch.load(model_path)
                if 'state_dict' in checkpoint.keys(): 
                    checkpoint = checkpoint['state_dict']
                from collections import OrderedDict
                new_checkpoint = OrderedDict()
                for k, v in checkpoint.items():
                    namekey = k[7:]
                    new_checkpoint[namekey] = v
                self.model.load_state_dict(new_checkpoint)
                

        else:
            self._net_init()


    def get_current_visual(self, need_np=True, need_HR=True):
        """
        return mosaic demosaic (ground truth) images
        """
        out_dict = OrderedDict()
        out_dict['mosaic'] = self.mosaic.data[0].float().cpu()
        out_dict['demosaic'] = self.demosaicked.data[0].float().cpu()
        if need_np:
            out_dict['mosaic'], out_dict['demosaic'] = util.tensorToNumpy([out_dict['mosaic'], out_dict['demosaic']])
        if need_HR:
            out_dict['ground_truth'] = self.ground_truth.data[0].float().cpu()
            if need_np: 
                out_dict['ground_truth'] = util.tensorToNumpy([out_dict['ground_truth']])[0]
        return out_dict


    def save_current_visual(self, epoch, iter, fpath):
        """
        save visual results for comparison
        """
        if epoch % self.save_vis_step == 0:
            visuals = self.get_current_visual(need_np=False)
            visuals_list=[util.quantize(visuals['ground_truth'].squeeze(0)),
                                 util.quantize(visuals['demosaic'].squeeze(0))]
            # visuals_list=[visuals['ground_truth'].squeeze(0), visuals['demosaic'].squeeze(0)]
            visual_images = torch.stack(visuals_list)
            visual_images = torchvision.utils.make_grid(visual_images, nrow=2, padding=5)
            # print(visual_images.dtype)
            visual_images = visual_images.byte().permute(1, 2, 0).numpy()
            visual_images = cv2.cvtColor(visual_images,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.visual_dir, 'epoch_{}_{}'.format(epoch, os.path.basename(fpath))),
                        visual_images)


    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']


    def update_learning_rate(self):
        self.scheduler.step()


    def get_current_log(self):
        log = OrderedDict()
        log['epoch'] = self.cur_epoch
        log['best_pred'] = self.best_pred
        log['best_epoch'] = self.best_epoch
        log['records'] = self.records
        return log


    def set_current_log(self, log):
        self.cur_epoch = log['epoch']
        self.best_pred = log['best_pred']
        self.best_epoch = log['best_epoch']
        self.records = log['records']


    def save_current_log(self):
        data_frame = pd.DataFrame(
            data={'train_loss': self.records['train_loss']
                , 'val_loss': self.records['val_loss']
                , 'psnr': self.records['psnr']
                , 'ssim': self.records['ssim']
                , 'lr': self.records['lr']
                  },
            index=range(1, self.cur_epoch + 1)
        )
        data_frame.to_csv(os.path.join(self.records_dir, 'train_records.csv'),
                          index_label='epoch')


    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                                 self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.ckpt_root, 'network_summary.txt'), 'w') as f:
                f.writelines(net_lines)

        print("==================================================")
