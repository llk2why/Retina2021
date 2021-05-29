import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

plt.switch_backend('agg')

cfa_order = [
    'Random_base',
    'Random_pixel',
    'RandomFuse2',
    'RandomFuse3',
    'RandomFuse4',
]

savedir = 'hierarchy_cross_b=0.0200'

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
    datasets = ["vdp","moire","McM","Kodak"]
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
    save_path = savedir+'/retina_result_a={}_b={}_{}.png'.format(a,b,date)
    plt.savefig(save_path,dpi=300)

    with open(savedir+'/retina_result_a={}_b={}_{}.txt'.format(a,b,date),'w') as f:
        f.write('{:<15}'.format(''))
        for dataset in datasets:
            f.write('&{:^15}'.format(dataset))
        f.write('\\\\\n')
        f.write('\\midrule\n')
        for i,cfa in enumerate(cfa_order):
            cfa_italic = '\\textit{{{}}}'.format(cfa.replace('_','\\_'))
            f.write('& {:<23}'.format(cfa_italic))
            for dataset in datasets:
                metric = '{:.2f}/{:.4f}'.format(psnrs[dataset][i],ssims[dataset][i])
                f.write('&{:^15}'.format(metric))
            f.write('\\\\\n')
    return {'ssim':ssims,'psnr':psnrs}
dir_patterns = glob.glob('../results/hierarchy/hierachy_b=0.0200/*')
dir_patterns.sort()

total_result = {}
for dir_pattern in dir_patterns:
    result_txts = glob.glob(dir_pattern+'/*/result.txt')
    result_txts.sort()
    a = re.findall(r"a=(.+?)_",dir_pattern)[0]
    b = re.findall(r"0_b=(.+?)$",dir_pattern)[0]
    key = a+b
    result = gather_result(result_txts,a,b)
    total_result[key] = result
print(total_result.keys())
json.dump(total_result,open(savedir+'/result.json','w'),indent=2)
a = 0
datasets = ['Kodak', 'McM', 'moire', 'vdp']

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/usr/share/fonts/winfonts/winfonts/simhei.ttf")
plt.rcParams.update({'font.size': 18})
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
colors = ['skyblue','cyan','lightseagreen','lightslategrey']
for i,cfa in enumerate(cfa_order):
    plt.clf()
    for dataset,color in zip(datasets,colors):
        psnrs = []
        b_noises = [0,0.005,0.01,0.02,0.03,0.04]
        for b in b_noises:
            key = '{:.4f}{:.4f}'.format(a,b)
            psnr = total_result[key]['psnr'][dataset][i]
            psnrs.append(psnr)
        plt.plot(b_noises,psnrs,marker='x',color=color)
    plt.legend(datasets)
    plt.xlabel(r'$\beta$')
    plt.ylabel('PSNR(dB)')
    plt.ylim(15,45)
    # title = r'$\it{'+cfa+r'}'+u'$ 采样'
    # title = title.replace('_',r'\_')
    # plt.title(title,fontproperties=font)
    plt.savefig(savedir+f'/b=0.0200_{cfa}_noise_tendency.png',dpi=300,bbox_inches='tight')
    
