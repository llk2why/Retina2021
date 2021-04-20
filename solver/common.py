import functools

import torch
import torch.nn as nn

from torch.nn import init

""" #################### """
"""      initialize      """
"""       network        """
""" #################### """

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [{}] ...'.format(classname))
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [{}] ...'.format(classname))
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.constant_(m.weight.data, 1.0)
        m.weight.data *= scale
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [{}] ...'.format(classname))
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [{}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_type))

""" #################### """
"""       create         """
"""       network        """
""" #################### """

# choose one network
def create_model(opt):

    model = opt['network']
    print('===> Building network [{}]...'.format(model))

    # demosaic models
    net = None
    if model in [
            'DeepJoint',
            'JointPixel',
            'JointPixelX',
            'JointPixelMax',
            'JointPixelMin',
            'JointPixelInception',
            'DeepResidual',
            'UNet',
            'DoublePath'
        ]:
        params = ''
        if 'RandomFuse' in opt['cfa'] or 'RandomBaseFuse' in opt['cfa']:
            fusion_degrees = {
                'RandomFuse2':2,
                'RandomFuse3':3,
                'RandomFuse4':4,
                'RandomFuse6':4,
                'RandomBaseFuse2':2,
                'RandomBaseFuse3':3,
                'RandomBaseFuse4':4,
                'RandomBaseFuse6':4,
            }
            if opt['cfa'] not in fusion_degrees:
                raise ValueError('not supported cfa:{}'.format(opt['cfa']))
            params = 'fusion_degree={}'.format(fusion_degrees[opt['cfa']])
            model = model + '_fusion'
        exec('from network import {}'.format(model))
        print(model)
        net = eval('{}({})'.format(model,params))
    else:
        raise NotImplementedError("Network [{}] is not recognized.".format(model))

    return net