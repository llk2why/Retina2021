import numpy as np

def _process_capsule(mosaic,bind_pattern,mask,capsule_type):
    index = bind_pattern==capsule_type
    assert mosaic.dtype == np.float32
    type_loc_delta = {
        1:[],
        2:[(0,1)],
        3:[(1,0)],
        4:[(0,1),(1,1)],
        5:[(1,0),(1,1)],
        6:[(1,0),(0,1)],
        7:[(-1,0),(0,-1)],
        8:[(0,1),(0,2)],
        9:[(1,0),(2,0)],
        10:[(0,1),(1,0),(1,1)],
        11:[(1,0),(2,0),(2,1)],
        12:[(1,0),(2,0),(3,0)],
        13:[(0,1),(0,2),(0,3)],
        14:[(0,1),(1,1),(1,2)]
    }
    index_list = []
    sum = np.zeros_like(mosaic).astype(np.float32)
    sum[index] = mosaic[index]

    total_idx = index.copy()
    for di,dj in type_loc_delta[capsule_type]:
        idx = index.copy()
        if di!=0: idx = np.concatenate((idx[-di:,:],idx[:-di,:]),axis=0)
        if dj!=0: idx = np.concatenate((idx[:,-dj:],idx[:,:-dj]),axis=1)
        sum[index] += mosaic[idx]
        total_idx = total_idx | idx
        index_list.append(idx)
    sum = np.sum(sum,axis=2)
    sum[index] = sum[index]/(len(index_list)+1)
    for idx in index_list:
        sum[idx] = sum[index]
    mosaic[total_idx] = mask[total_idx]*(sum[:,:,None][total_idx])


def check_capsule(mosaic,bind_pattern,mask):
    type_loc_delta = {
        1:[],
        2:[(0,1)],
        3:[(1,0)],
        4:[(0,1),(1,1)],
        5:[(1,0),(1,1)],
        6:[(1,0),(0,1)],
        7:[(-1,0),(0,-1)],
        8:[(0,1),(0,2)],
        9:[(1,0),(2,0)],
        10:[(0,1),(1,0),(1,1)],
        11:[(1,0),(2,0),(2,1)],
        12:[(1,0),(2,0),(3,0)],
        13:[(0,1),(0,2),(0,3)],
        14:[(0,1),(1,1),(1,2)]
    }
    h,w,c = mosaic.shape
    sum = np.sum(mosaic,axis=2)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                if abs(mosaic[i,j,k])>1e-6 and mask[i,j,k]==0:
                    return False
    for i in range(h):
        for j in range(w):
            if bind_pattern[i,j] not in type_loc_delta: continue
            for di,dj in type_loc_delta[bind_pattern[i,j]]:
                ii,jj = i+di,j+dj
                if(sum[ii][jj]!=sum[i][j]):
                    tmp = mosaic[i:i+3,j:j+3]
                    print(i,j,'\t',ii,jj)
                    print(tmp[...,0])
                    print(tmp[...,1])
                    print(tmp[...,2])
                    print(bind_pattern[i:i+3,j:j+3])
                    print(sum[i,j],sum[ii,jj])
                    print(bind_pattern[i,j])
                    return False
    return True


def check_pattern(name):
    if name=='Random_pixel': return
    elif name=='Random_2JCS': capsule_types = [2,3]
    elif name=='Random_3JCS': capsule_types = [4,5,6,7,8,9]
    elif name=='Random_4JCS': capsule_types = [10,11,12,13,14]
    base_pattern = np.load('../cfa_pattern/random_base128.npy')
    bind_pattern = np.load('../cfa_pattern/{}.npy'.format(name))
    mosaic = np.random.randint(1,10,size=(128,128,3)).astype(np.float32)
    mask = base_pattern*((bind_pattern!=0)[:,:,None])
    mosaic = mask*mosaic
    # tmp = base_pattern[:4,:4]
    # print('\nbase_pattern')
    # print(tmp[...,0])
    # print(tmp[...,1])
    # print(tmp[...,2])
    # tmp = mosaic[:4,:4]
    # print('\n initialized mosaic')
    # print(tmp[...,0])
    # print(tmp[...,1])
    # print(tmp[...,2])
    for capsule_type in capsule_types:
        _process_capsule(mosaic,bind_pattern,mask,capsule_type)
    # print('\nprocessed mosaic')
    # tmp = mosaic[:4,:4]
    # print(tmp[...,0])
    # print(tmp[...,1])
    # print(tmp[...,2])
    # print(bind_pattern[:4,:4])
    # exit()
    result = check_capsule(mosaic,bind_pattern,mask)
    if result: print('{} passed test\n\n'.format(name))
    else: print('{} failed test\n\n'.format(name))


def main():
    random_type_class = {
        'Random_2JCS':2,
        'Random_3JCS':3,
        'Random_4JCS':4
    }
    for name in random_type_class:
        check_pattern(name)

if __name__ == '__main__':
    main()