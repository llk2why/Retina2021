import os
import glob
import numpy as np
import matplotlib.pyplot as plt


cfa_order = [
    # 'Random_6JCS',
    # 'Random_4JCS',
    # 'Random_3JCS',
    # 'Random_2JCS',
    # '4JCS',
    # '3JCS',
    # '2JCS',
    # 'Random_pixel',
    # 'RandomFuse2',
    # 'RandomFuse3',
    # 'RandomFuse4',
    # 'RandomFuse6',
    'RandomBaseFuse2',
    'RandomBaseFuse3',
    'RandomBaseFuse4',
    'RandomBaseFuse6',
    'Random_base',
    # 'RGGB'
]

a = 0.0
b = 0.04

result_txts = glob.glob('../results/001_JOINTPIXEL_on_MIT_a={:.4f}_b={:.4f}/*/*.txt'.format(a,b))

def parse_result(txt_path):
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

def gather_result():
    results = {}
    for result_txt in result_txts:
        cfa,res = parse_result(result_txt)
        results[cfa] = res
    # label_list = ['R4','R3','R2','2JCS','R1','RF2','RF3','RF4','RGGB']
    # label_list = ['Random_4JCS','Random_3JCS','Random_2JCS','2JCS','Random_pixel','RandomFuse2','RandomFuse3','RandomFuse4','RGGB']
    label_list = cfa_order
    kodak_psnr = [float(results[cfa]["Kodak"].split('/')[0]) for cfa in cfa_order]
    mcm_psnr = [float(results[cfa]["McM"].split('/')[0]) for cfa in cfa_order]
    moire_psnr = [float(results[cfa]["moire"].split('/')[0]) for cfa in cfa_order]
    mit_psnr = [float(results[cfa]["MIT"].split('/')[0]) for cfa in cfa_order]
    x = np.arange(len(label_list))*10
    width = 3.5

    fig, ax = plt.subplots()
    ax.set_ylim(23,50)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',fontsize=6)
    psnrs = {
        'Kodak':kodak_psnr,
        'McM':mcm_psnr,
        'moire':moire_psnr,
        'MIT':mit_psnr
    }
    colors = ['lightsteelblue','lightblue','lightseagreen','lightslategrey']
    for i,dataset in enumerate(["MIT","moire","McM","Kodak"]):
        rect = ax.bar(x - (width*3/4)+(width*i/2), psnrs[dataset], width/2, label=dataset,color=colors[i])
        autolabel(rect)

    ax.set_ylabel('PSNR')
    ax.set_title('Retinal Experiment results')
    ax.set_xticks(x)
    ax.set_xticklabels(label_list,rotation=70)
    ax.legend()

    fig.tight_layout()
    import datetime
    date = datetime.datetime.now().strftime('%Y%m%d')
    plt.savefig('retina_result_a={:.4f}_b={:.4f}_{}.png'.format(a,b,date))

gather_result()

    




