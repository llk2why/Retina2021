import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

plt.switch_backend('agg')

cfa_order = [
    'RGGB',
    '2JCS',
    '3JCS',
    '4JCS',
]

savedir = 'JCS'


os.makedirs(savedir,exist_ok=True)

def parse_result(txt_path,a,b):
    result = {}
    cfa = txt_path.split('/')[-2]
    with open(txt_path,'r') as f:
        f.readline()
        line = f.readline()
        while line:
            line = line.strip()
            dataset,metric = line.split(':')
            if dataset == 'MyImage':
                dataset = 'Sandwich'
            if dataset == 'MIT':
                dataset = 'vdp'
            result[dataset] = metric
            line = f.readline()
    return cfa,result

def gather_result(result_txts,a,b):
    results = {}
    for result_txt in result_txts:
        print(result_txt)
        cfa,res = parse_result(result_txt,a,b)
        results[cfa] = res


    label_list = cfa_order
    for cfa in cfa_order:
        print(cfa,results[cfa].keys())
        results[cfa]["McM"]
    datasets = ["vdp","moire","McM","Kodak","Sandwich"]
    psnrs = OrderedDict()
    ssims = OrderedDict()
    for dataset in datasets:
        psnrs[dataset] = [float(results[cfa][dataset].split('/')[0]) for cfa in cfa_order]
        ssims[dataset] = [float(results[cfa][dataset].split('/')[1]) for cfa in cfa_order]
    
    x = np.arange(len(label_list))*10
    width = 3.5

    fig, ax = plt.subplots()
    ax.set_ylim(15,50)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',fontsize=6)

    colors = ['lightsteelblue','lightblue','lightseagreen','lightslategrey','blue']
    for i,dataset in enumerate(datasets):
        rect = ax.bar(x - (width*3/4)+(width*i/2), psnrs[dataset], width/2, label=dataset,color=colors[i])
        autolabel(rect)

    ax.set_ylabel('PSNR')
    ax.set_title('Retinal Experiment results')
    ax.set_xticks(x)
    ax.set_xticklabels(label_list,rotation=70)
    ax.legend(loc='upper left')

    fig.tight_layout()
    import datetime
    date = datetime.datetime.now().strftime('%Y%m%d')
    plt.savefig(savedir+'/retina_result_a={}_b={}_{}.png'.format(a,b,date),dpi=300)

    with open(savedir+'/retina_result_a={}_b={}_{}.txt'.format(a,b,date),'w') as f:
        f.write('{:<15}'.format(''))
        for dataset in datasets:
            f.write('&{:^15}'.format(dataset))
        f.write('\\\\\n')
        f.write('\\midrule\n')
        for i,cfa in enumerate(cfa_order):
            cfa_italic = '\\textit{{{}}}'.format(cfa.replace('_','\\_').replace('RGGB','Bayer'))
            f.write('& {:<15}'.format(cfa_italic))
            for dataset in datasets:
                metric = '{:.2f}/{:.4f}'.format(psnrs[dataset][i],ssims[dataset][i])
                f.write('&{:^15}'.format(metric))
            if 'Sandwich' not in datasets:
                f.write('&{:^15}'.format('00.00/0.0000'))
            f.write('\\\\\n')
    return {'ssim':ssims,'psnr':psnrs}

b_noises = [0.,0.01,0.02,0.03,0.04]
a_noises = [0.,0.01,0.02,0.03,0.04]
noise_pairs = [(0,b) for b in b_noises]
noise_pairs += [(a,0.0016) for a in a_noises]
for a,b in noise_pairs:
    total_result = {}
    b = '{:.4f}'.format(b)
    a = '{:.4f}'.format(a)
    print(a,b)
    dir_pattern = '../results/jcs/001_JOINTPIXEL_on_MIT_a={}_b={}'.format(a,b)


    result_txts = glob.glob(dir_pattern+'/*/result.txt')
    result_txts.sort()
    result_txts = [x for x in result_txts if any([key in x for key in cfa_order])]

    a = re.findall(r"a=(.+?)_",dir_pattern)[0]
    b = re.findall(r"b=(.+?)$",dir_pattern)[0]
    key = a+b
    result = gather_result(result_txts,a,b)
    # total_result[key] = result

