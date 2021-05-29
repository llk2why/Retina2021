import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

plt.switch_backend('agg')

cfa_order = [
    'Random_base',
    'RandomBaseFuse2',
    'RandomBaseFuse3',
    'RandomBaseFuse4',
]

os.makedirs('hierarchy_base_cross',exist_ok=True)

def parse_result(txt_path,a,b):
    result = {}
    cfa = txt_path.split('/')[-2]
    with open(txt_path,'r') as f:
        f.readline()
        line = f.readline()
        while line:
            line = line.strip()
            dataset,metric = line.split(':')
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
    datasets = ["MIT","moire","McM","Kodak"]
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

    colors = ['lightsteelblue','lightblue','lightseagreen','lightslategrey']
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
    plt.savefig('hierarchy_base_cross/retina_result_a={}_b={}_{}.png'.format(a,b,date),dpi=300)

    with open('hierarchy_base_cross/retina_result_a={}_b={}_{}.txt'.format(a,b,date),'w') as f:
        f.write('{:<15}'.format(''))
        for dataset in datasets:
            f.write('&{:^15}'.format(dataset))
        f.write('\\\\\n')
        f.write('\\midrule\n')
        for i,cfa in enumerate(cfa_order):
            f.write('{:<15}'.format(cfa.replace('_','\\_')))
            for dataset in datasets:
                metric = '{:.2f}/{:.4f}'.format(psnrs[dataset][i],ssims[dataset][i])
                f.write('&{:^15}'.format(metric))
            f.write('\\\\\n')
dir_patterns = glob.glob('../results/hierarchy/base_noise/*')
dir_patterns.sort()
for dir_pattern in dir_patterns:
    result_txts = glob.glob(dir_pattern+'/*/result.txt')
    result_txts.sort()
    a = re.findall(r"a=(.+?)_",dir_pattern)[0]
    b = re.findall(r"b=(.+?)$",dir_pattern)[0]

    gather_result(result_txts,a,b)

    




