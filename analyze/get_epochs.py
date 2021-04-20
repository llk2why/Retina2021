import os
import glob
import pandas as pd

def extract_epoch(fpath):
    df = pd.read_csv(fpath)
    epoch = df.tail(1)['epoch'] 
    cfa = fpath.split('/')[2]
    print(cfa,int(epoch))

def walk_csv(a,b):
    pattern = '../checkpoints/*/001_JointPixel_MIT_a={:.4f}_b={:.4f}/records/train_records.csv'
    pattern = pattern.format(a,b)
    csv_paths = glob.glob(pattern)
    print('a={:.4f}_b={:.4f}'.format(a,b))
    for fpath in csv_paths:
        extract_epoch(fpath)


if __name__ == '__main__':
    walk_csv(a=0.0,b=0.0)