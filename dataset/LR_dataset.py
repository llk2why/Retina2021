import torch.utils.data as data

from dataset import common


class LRDataset(data.Dataset):
    '''
    Read Mosaic images of ground truth in the test phase.
    '''

    def name(self):
        return common.find_benchmark(self.opt['datadir'])


    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.paths_ground_truth = common.get_image_paths(self.opt['datadir'])

        assert self.paths_ground_truth, '[Error] The ground truth set are empty.'


    def __getitem__(self, idx):
        mosaic, ground_truth, ground_truth_path = self._load_file(idx)
        mosaic_tensor = common.ToTensor_([mosaic])[0]
        return {'mosaic': mosaic_tensor, 'ground_truth_path': ground_truth_path}


    def __len__(self):
        return len(self.paths_ground_truth)


    def _load_file(self, idx):
        ground_truth_path = self.paths_ground_truth[idx]
        ground_truth = common.read_img(ground_truth_path)
        if self.opt['cfa_type'] == 'bayer' or self.opt['cfa_type'] is None:
            mosaic = common.bayer(ground_truth)
        else:
            raise ValueError('Unsupported CFA type')
        return mosaic, ground_truth, ground_truth_path
