import os
import cv2
import sys
import numpy as np


from latex_tools import head,end,draw_color_unit,draw_transparent_circle


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
    elif type == 6:
        draw_code[(draw_code<=60)&(draw_code!=-6)]=-1

    # 混色
    color_ = color.copy()
    for i in range(h):
        for j in range(w):
            code = draw_code[i][j]
            queue = [(i,j)]
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
                14:[(0,1),(1,1),(1,2)],
                61:[(0,1),(1,0),(1,1),(2,0),(2,1)],
                62:[(-1,0),(-1,1),(0,-1),(0,1),(0,2)],
                63:[(-1,0),(0,-1),(0,1),(0,2),(1,1)],
                64:[(1,0),(1,1),(1,2),(1,3),(2,2)],
                65:[(0,1),(1,0),(1,1),(1,2),(2,1)],
                66:[(0,1),(1,1),(2,1),(2,2),(2,3)],
                67:[(1,0),(2,0),(3,0),(3,1),(3,2)],
                68:[(0,1),(0,2),(0,3),(0,4),(0,5)],
                69:[(0,1),(1,1),(2,1),(3,1),(3,2)],
            }
            if code in type_loc_delta:
                for di,dj in type_loc_delta[code]:
                    ii = i+di
                    jj = j+dj
                    # print(ii,jj,end=' ')
                    if type == 5 or \
                    (code in range(2,4) and type == 2) or \
                    (code in range(4,10) and type == 3) or \
                    (code in range(10,15) and type ==4 ) or \
                    (code in range(61,70) and type ==6 ):
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

    draw_transparent_circle(0,0,'R')
    draw_transparent_circle(h-1,w-1,'R')
    # for i in range(h):
    #     for j in range(w):
    #         tex_ts += " \\fill[R,draw, shift = {}, opacity=0.0]" .format("{(" + str(j) +", " + str(-i) + ")}")
    #         tex_ts += "{[rounded corners=9](-0.37, -0.37) -- (0.37, -0.37)  -- (0.37,0.37) -- (-0.37,0.37) -- cycle {}};\n"
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

    info = {
        2:'Random_2JCS',
        3:'Random_3JCS',
        4:'Random_4JCS',
        6:'Random_6JCS',
    }

    if capsule_class == 1:
        name = 'Random_pixel'
        draw_code = np.ones((128,128))
    elif capsule_class in info:
        name = info[capsule_class]
        draw_code = np.load('../{}.npy'.format(name))
    else:
        raise ValueError('unsupported capsule class!')

    return cfa,draw_code,name
    
    


def main():
    os.makedirs('hierarchy_cfa',exist_ok=True)
    # for i in range(1,6):
    for i in [6]:
        params = get_pattern(i)
        convert(params,i)


if __name__ == "__main__":
    main()

