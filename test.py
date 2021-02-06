import cv2
import tqdm
import torch
# import imageio
import argparse, time, os
import options.option as option

from utils import util
from solver import Solver
from dataset import LRDataset,LRHRDataset


def parse_args():
    parser = argparse.ArgumentParser(description='frequent testing parameters')
    parser.add_argument('--opt', type=str, required=True, help='Path to options JSON file')
    parser.add_argument('--is_train', action='store_true', help='where to train')
    parser.add_argument('--save_image', action='store_true', help='whether to save validation images')
    parser.add_argument('--debug', action='store_true', help='debug switch')
    parser.add_argument('-a', type=float, default=0.0, help='Poisson noise parameter')
    parser.add_argument('-b', type=float, default=0.0, help='Gaussian noise parameter')
    parser.add_argument('--network',type=str,required=True,help='Network to use')
    parser.add_argument('--cfa',type=str,required=True,help='color filter array')
    parser.add_argument('--pretrained_path', type=str,required=True,help='Path to pretrained model for resume')
    return parser.parse_args()


def main():
    args = parse_args()
    opt = option.parse(args)

    # initial configure
    model_name =  opt['network'].upper()
    if opt['self_ensemble']: 
        model_name += 'plus'

    # create test dataloader
    bm_names =[]
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        dataset_opt['cfa'] = opt['cfa']
        dataset_opt['a'] = opt['a']
        dataset_opt['b'] = opt['b']
        test_dataset = LRHRDataset(dataset_opt)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=10, 
            pin_memory=True
        )
        test_loaders.append(test_loader)
        print('===> Test Dataset: [{}]   Number of images: [{}]'.format(test_dataset.name(), len(test_dataset)))
        bm_names.append(test_dataset.name())

    # create solver (and load model)
    solver = Solver(opt)

    # Test phase
    print('===> Start Test')
    print("==================================================")
    print("Method: {}".format(model_name))

    psnrs = []
    ssims = []
    model_dataset_name = model_name+'_on_{}'.format(opt['dataset_name'])
    for bm, test_loader in zip(bm_names, test_loaders):
        print("Test set : [{}]".format(bm))

        sr_list = []
        path_list = []

        total_psnr = []
        total_ssim = []
        total_time = []

        print(test_loader.dataset.__class__.__name__)
        need_HR = False if test_loader.dataset.__class__.__name__.find('LRHR') < 0 else True

        for iter, batch in enumerate(test_loader):
            print(batch['ground_truth_path'])
            solver.feed_data(batch, need_HR=need_HR)

            # calculate forward time
            t0 = time.time()
            solver.test()
            t1 = time.time()
            total_time.append((t1 - t0))

            visuals = solver.get_current_visual(need_HR=need_HR)
            sr_list.append(visuals['demosaic'])
            
            # output_mosaic = cv2.cvtColor(visuals['mosaic'],cv2.COLOR_RGB2BGR)
            # cv2.imwrite('tmp/{}.png'.format(iter),output_mosaic)

            # calculate PSNR/SSIM metrics on Python
            if need_HR:
                # im_target = imageio.imread(batch['ground_truth_path'][0],pilmode='RGB')
                im_target = cv2.imread(batch['ground_truth_path'][0])
                im_target = cv2.cvtColor(im_target,cv2.COLOR_BGR2RGB)
                psnr, ssim = util.calc_metrics(visuals['demosaic'], visuals['ground_truth'])
                total_psnr.append(psnr)
                total_ssim.append(ssim)
                path_list.append(os.path.basename(batch['ground_truth_path'][0]).replace('ground_truth', model_name))
                print("[%d/%d] %s || PSNR(dB)/SSIM: %.2f/%.4f || Timer: %.4f sec ." % (iter+1, len(test_loader),
                                                                                       os.path.basename(batch['ground_truth_path'][0]),
                                                                                       psnr, ssim,
                                                                                       (t1 - t0)))
                # im_target = imageio.imread(batch['ground_truth_path'][0])
                tmp_psnr = util.calc_psnr(visuals['demosaic'],im_target)
                print('new PSNR: {:.2f} delta:{:.2f}'.format(tmp_psnr,tmp_psnr-psnr))
            else:
                path_list.append(os.path.basename(batch['ground_truth_path'][0]))
                print("[%d/%d] %s || Timer: %.4f sec ." % (iter + 1, len(test_loader),
                                                           os.path.basename(batch['ground_truth_path'][0]),
                                                           (t1 - t0)))

        if need_HR:
            print("---- Average PSNR(dB) /SSIM /Speed(s) for [%s] ----" % bm)
            print("PSNR: %.2f      SSIM: %.4f      Speed: %.4f" % (sum(total_psnr)/len(total_psnr),
                                                                  sum(total_ssim)/len(total_ssim),
                                                                  sum(total_time)/len(total_time)))
            psnrs.append(sum(total_psnr)/len(total_psnr))
            ssims.append(sum(total_ssim)/len(total_ssim))
        else:
            print("---- Average Speed(s) for [%s] is %.4f sec ----" % (bm,
                                                                      sum(total_time)/len(total_time)))

        # save demosaic results for further evaluation on MATLAB
        if need_HR:
            save_img_path = os.path.join('./results/demosaic', model_dataset_name, bm)
        else:
            save_img_path = os.path.join('./results/demosaic', model_dataset_name,bm)
        if opt['output_dir'] is not None:
            save_img_path=save_img_path.replace('./results/demosaic',opt['output_dir'])

        print("===> Saving demosaic images of [%s]... Save Path: [%s]\n" % (bm, save_img_path))

        if not os.path.exists(save_img_path): 
            os.makedirs(save_img_path)
        
        for img, name in tqdm.tqdm(zip(sr_list, path_list)):
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_img_path, name), img)
            # imageio.imwrite(os.path.join(save_img_path, name), img)

    if need_HR:
        if opt['mode']=='demosaic':
            result_txt_path = os.path.join('./results/demosaic', model_dataset_name,'result.txt')
            if opt['output_dir'] is not None:
                result_txt_path = os.path.join(opt['output_dir'], model_dataset_name,'result.txt')
            with open(result_txt_path,'w') as f:
                f.write('{}\n'.format(model_dataset_name.replace('_',' ')))
                for bm,psnr,ssim in zip(bm_names,psnrs,ssims):
                    f.write('{}:{:.2f}/{:.4f}\n'.format(bm,psnr,ssim))
    print("==================================================")
    print("===> Finished !")

if __name__ == '__main__':
    main()