import os
import cv2
import sys
import numpy as np


from latex_tools import head,end,draw_color_unit


def convert(parms, type):
    color,draw_code,name = parms
    output = "hierarchy_cfa/" + name + ".tex"

    color = color.copy().astype('str')
    draw_code = draw_code.copy()
    color[color=='0'] = 'R'
    color[color=='1'] = 'G'
    color[color=='2'] = 'B'
    tex_ts = ""

    h,w = draw_code.shape

    if type == 0:
        draw_code[draw_code!=1]=1
    elif type == 1:
        draw_code[draw_code!=1]=-1
    elif type == 2:
        draw_code[(draw_code!=2)&(draw_code!=3)&(draw_code!=-2)&(draw_code!=-3)]=-1
    elif type == 3:
        draw_code[((draw_code<4)|(draw_code>9))&(draw_code!=-3)]=-1
    elif type == 4:
        draw_code[((draw_code<10)|(draw_code>14))&(draw_code!=-4)]=-1

    # 混色
    color_ = color.copy()
    for i in range(h):
        for j in range(w):
            code = draw_code[i][j]
            queue = [(i,j)]
            op = {
                2:[1],
                3:[128],
                4:[1,129],
                5:[128,129],
                6:[1,128],
                7:[-1,-128],
                8:[1,2],
                9:[128,256],
                10:[1,128,129],
                11:[128,256,257],
                12:[128,256,384],
                13:[1,2,3],
                14:[1,129,130]
            }
            if code in op:
                for idx in op[code]:
                    ii = i+(j+idx)//128
                    jj = (j+idx)%128
                    # print(ii,jj,end=' ')
                    if type == 5 or \
                    (code in range(2,4) and type == 2) or \
                    (code in range(4,10) and type == 3) or \
                    (code in range(10,15) and type ==4 ):
                        queue.append((ii,jj))
                        if ii>=128 or jj>=128:
                            print('\nBUG!')
                            print(i,j)
                            print(ii,jj)
                            print(type)
                            exit()
                    
            palette = ''
            # print(queue,code)
            if len(queue) > 1:
                for x,y in queue:
                    # print(x,y)
                    palette += color_[x][y]
                palette = ''.join(sorted(palette))
                for x,y in queue:
                    color[x][y] = palette
        
    
    # print(draw_code)
    # print(color)

    for i in range(h):
        for j in range(w):
            tex_ts += " \\fill[R,draw, shift = {}, opacity=0.0]" .format("{(" + str(j) +", " + str(-i) + ")}")
            tex_ts += "{[rounded corners=9](-0.37, -0.37) -- (0.37, -0.37)  -- (0.37,0.37) -- (-0.37,0.37) -- cycle {}};\n"
    for i in range(h):
        for j in range(w):
            tp = draw_code[i][j]
            if tp<0:
                continue
            else:
                tex_ts += draw_color_unit(tp,color[i][j],i,j,h,w)

            


    with open(output,"w",encoding='utf8') as fp:
        fp.write(head + tex_ts +end)


def get_pattern(capsule_class):
    flatten_pattern = np.load('../random_base128.npy')
    flatten_pattern = cv2.cvtColor(flatten_pattern,cv2.COLOR_BGR2RGB)
    cfa = np.argmax(flatten_pattern,axis=2)

    if capsule_class == 1:
        draw_code = np.ones((128,128))
        name = 'Random_pixel'
    elif capsule_class == 2:
        draw_code = np.load('../Random_2JCS.npy')
        name = 'Random_2JCS'
    elif capsule_class == 3:
        draw_code = np.load('../Random_3JCS.npy')
        name = 'Random_3JCS'
    elif capsule_class == 4:
        draw_code = np.load('../Random_4JCS.npy')
        name = 'Random_4JCS'
    
    return cfa,draw_code,name
    
    


def main():
    os.makedirs('hierarchy_cfa',exist_ok=True)
    for i in range(1,5):
        params = get_pattern(i)
        convert(params,i)


if __name__ == "__main__":
    main()

