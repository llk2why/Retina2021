import os
import sys
import tqdm
import torch
import random
import argparse
import options.option as option
import torch.multiprocessing as mp
import torch.distributed as dist

from utils import *
from solver import Solver
from torch.utils.tensorboard import SummaryWriter
from dataset import LRDataset,LRHRDataset

def init(rank, world_size, opt):
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True
    torch.cuda.set_device(rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')

    # redirect output of process i>0 to nulldev
    if(rank!=0):
        sys.stdout = open(os.devnull,'w')


def cleanup():
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='frequent training parameters')
    parser.add_argument('--opt', type=str, required=True, help='Path to options JSON file')
    parser.add_argument('--is_train', action='store_false', help='where to train')
    parser.add_argument('--save_image', action='store_true', help='whether to save validation images')
    parser.add_argument('--debug', action='store_true', help='debug switch')
    parser.add_argument('-a', type=float, default=0.0, help='Poisson noise parameter')
    parser.add_argument('-b', type=float, default=0.0, help='Gaussian noise parameter')
    parser.add_argument('--network',type=str,required=True,help='Network to use')
    parser.add_argument('--cfa',type=str,required=True,help='color filter array')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model for resume')
    parser.add_argument('--port', type=str, default="12355", help='DDP communication port')
    return parser.parse_args()


def init_dataloader(opt):
    # create train and val dataloader
    world_size = opt['world_size']
    rank = opt['rank']
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        dataset_opt['cfa'] = opt['cfa']
        dataset_opt['a'] = opt['a']
        dataset_opt['b'] = opt['b']
        if phase == 'train':
            train_dataset = LRHRDataset(dataset_opt)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=dataset_opt['batch_size'],
                shuffle=False, 
                num_workers=dataset_opt['n_workers'], 
                pin_memory=True,
                sampler=train_sampler
            )
            print('===> Train Dataset: {}   Number of images: [{}]'.format(train_dataset.name(), len(train_dataset)))
            assert train_loader is not None, '[Error] The training data does not exist' 
        elif phase == 'val':
            val_dataset = LRHRDataset(dataset_opt)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4, 
                pin_memory=True,
                sampler=val_sampler
            )
            print('===> Val Dataset: {}   Number of images: [{}]'.format(val_dataset.name(), len(val_dataset)))
            assert val_loader is not None, '[Error] The valing data does not exist' 
        else:
            raise ValueError('[Error] Dataset phase [%s] in *.json is not recognized.'.format(phase))
    return train_dataset,val_dataset,train_loader,val_loader


def main(rank,world_size):
    args = parse_args()
    init(rank, world_size, args)
    opt = option.parse(args,rank=rank,world_size=world_size)
    dist.barrier()
    train_dataset,val_dataset,train_loader,val_loader = init_dataloader(opt)

    solver = Solver(opt)
    model_name = opt['network'].upper()

    print('===> Start Train')
    print("==================================================")

    solver_log = solver.get_current_log()

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']

    print("Method: {} || Noise a: {:.4f}  b: {:.4f} || Epoch Range: ({} ~ {})".format(
        model_name,
        opt['a'],
        opt['b'],
        start_epoch,
        NUM_EPOCH
    ))

    if rank==0:
        writer = SummaryWriter('logs/{}/{}_{}_{}_a={:.4f}_b={:.4f}'.format(
            opt['cfa'],
            str(opt['id']).zfill(3),
            opt['network'],
            opt['dataset_name'],
            opt['a'],
            opt['b']
        ))

    output_device = sys.stdout

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print('\n===> Training Epoch: [{}/{}]...  Learning Rate: {}'.format(epoch,
                                                                      NUM_EPOCH,
                                                                      solver.get_current_learning_rate()))

        # Initialization
        solver_log['epoch'] = epoch

        # Train model
        train_loss_list = []
        with tqdm.tqdm(total=len(train_loader), desc='Epoch: [{}/{}]'.format(epoch, NUM_EPOCH), miniters=1, file=output_device) as pbar:
            for iter, batch in enumerate(train_loader):
                solver.feed_data(batch)
                iter_loss = solver.train_step()
                batch_size = batch['mosaic'].size(0)
                train_loss_list.append(iter_loss*batch_size)
                if rank==0:
                    writer.add_scalar('Epochs/{}'.format(str(epoch).zfill(2)),iter_loss,iter)
                    writer.add_scalar('Total/Train Iter Loss',iter_loss,(epoch-1)*len(train_loader)+iter)
                pbar.set_postfix_str("Batch Loss: {:.4f}".format(iter_loss))
                pbar.update()
                if(opt['debug']):
                    if iter==10:
                        break

        epoch_loss = sum(train_loss_list)/len(train_dataset)
        solver_log['records']['train_loss'].append(epoch_loss)
        solver_log['records']['lr'].append(solver.get_current_learning_rate())

        print('\nEpoch: [{}/{}]   Avg Train Loss: {:.6f}'.format(epoch,
                                                    NUM_EPOCH,
                                                    epoch_loss))
                                                
        if rank==0:
            writer.add_scalar('Total/Train Loss',epoch_loss,epoch)

        print('===> Validating...')

        psnr_list = []
        ssim_list = []
        val_loss_list = []

        with tqdm.tqdm(total=len(val_loader), desc='Epoch(val): [{}/{}]'.format(epoch, NUM_EPOCH), miniters=1, file=output_device) as pbar:
            cnt = 0
            for iter, batch in enumerate(val_loader):
                solver.feed_data(batch)
                iter_loss = solver.test()
                batch_size = batch['mosaic'].size(0)
                val_loss_list.append(iter_loss*batch_size)
                

                # calculate evaluation metrics
                visuals = solver.get_current_visual()
                if rank==0 and epoch == 4:
                    permute = [2,1,0]
                    mosaic_output = batch['mosaic'][0][permute,:,:]
                    ground_truth_output = batch['ground_truth'][0][permute,:,:]
                    demosaicked_output = solver.demosaicked[0][permute,:,:]
                    writer.add_image('mosaic/{}_{}'.format(epoch,iter), mosaic_output)
                    writer.add_image('ground_truth/{}_{}'.format(epoch,iter), ground_truth_output)
                    writer.add_image('demosaicked/{}_{}'.format(epoch,iter), demosaicked_output)
                
                psnr, ssim = util.calc_metrics(visuals['ground_truth'], visuals['demosaic'])
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                if opt["save_image"]:
                    solver.save_current_visual(epoch, iter, batch['ground_truth_path'][0])
                if rank==0:
                    writer.add_scalar('Total/Val Iter Loss',iter_loss,(epoch-1)*len(val_loader)+iter)
                pbar.set_postfix_str("Batch Loss: {:.4f}".format(iter_loss))
                pbar.update()
                cnt += 1
                if(opt['debug']):
                    if cnt==10:
                        break
                
        if rank==0:
            val_loss = sum(val_loss_list)/len(val_loss_list)
            ave_psnr = sum(psnr_list)/len(psnr_list)
            ave_ssim = sum(ssim_list)/len(ssim_list)
            solver_log['records']['val_loss'].append(val_loss)
            solver_log['records']['psnr'].append(ave_psnr)
            solver_log['records']['ssim'].append(ave_ssim)

            writer.add_scalar('Total/Val Loss',val_loss,epoch)
            writer.add_scalar('Total/PSNR',ave_psnr,epoch)
            writer.add_scalar('Total/SSIM',ave_ssim,epoch)

            # record the best epoch
            epoch_is_best = False
            if solver_log['best_pred'] < (ave_psnr):
                solver_log['best_pred'] = (ave_psnr)
                epoch_is_best = True
                solver_log['best_epoch'] = epoch

            print("[{}] PSNR: {:.2f}   SSIM: {:.4f}   Loss: {:.6f}   Best PSNR: {:.2f} in Epoch: [{}]".format(val_dataset.name(),
                                                                                                ave_psnr,
                                                                                                ave_ssim,
                                                                                                val_loss,
                                                                                                solver_log['best_pred'],
                                                                                                solver_log['best_epoch']))

            solver.set_current_log(solver_log)
            solver.save_checkpoint(epoch, epoch_is_best)
            solver.save_current_log()

        # update lr
        solver.update_learning_rate()



    print('===> Finished !')
    
    
    cleanup()


def run():
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus # number of GPUs
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    run()