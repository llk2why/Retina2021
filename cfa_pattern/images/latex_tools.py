import os


head =r'''
\documentclass[tikz,border=1mm, convert={outfile=compiler.png}]{standalone}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}
\definecolor{id2}{rgb}{1,0,0}
\definecolor{id1}{rgb}{0,1,0}
\definecolor{id0}{rgb}{0,0,1}
\definecolor{Y}{rgb}{1,1,0}
\definecolor{M}{rgb}{1,0,1}
\definecolor{C}{rgb}{0,1,1}
\definecolor{Z}{rgb}{0,0,0}
\definecolor{W}{rgb}{1,1,1}
\definecolor{R}{rgb}{1,0,0}
\definecolor{G}{rgb}{0,1,0}
\definecolor{B}{rgb}{0,0,1}

\definecolor{BB}{rgb}{0,0,1}
\definecolor{BG}{rgb}{0,1,1}
\definecolor{BR}{rgb}{1,0,1}
\definecolor{GG}{rgb}{0,1,0}
\definecolor{GR}{rgb}{1,1,0}
\definecolor{RR}{rgb}{1,0,0}

\definecolor{BBB}{RGB}{0,0,255}
\definecolor{BBG}{RGB}{0,150,255}
\definecolor{BBR}{RGB}{150,0,255}
\definecolor{BGG}{RGB}{0,255,150}
\definecolor{BGR}{RGB}{150,255,150}
\definecolor{BRR}{RGB}{255,0,150}
\definecolor{GGG}{RGB}{0,255,0}
\definecolor{GGR}{RGB}{150,255,0}
\definecolor{GRR}{RGB}{255,150,0}
\definecolor{RRR}{RGB}{255,0,0}

\definecolor{BBBB}{RGB}{0,0,255}
\definecolor{BBBG}{RGB}{0,100,255}
\definecolor{BBBR}{RGB}{100,0,255}
\definecolor{BBGG}{RGB}{0,255,255}
\definecolor{BBGR}{RGB}{150,255,255}
\definecolor{BBRR}{RGB}{255,0,255}
\definecolor{BGGG}{RGB}{0,255,100}
\definecolor{BGGR}{RGB}{150,255,150}
\definecolor{BGRR}{RGB}{255,150,150}
\definecolor{BRRR}{RGB}{255,0,100}

\definecolor{GGGG}{RGB}{0,255,0}
\definecolor{GGGR}{RGB}{100,255,0}
\definecolor{GGRR}{RGB}{255,255,0}
\definecolor{GRRR}{RGB}{255,100,0}
\definecolor{RRRR}{RGB}{255,0,0}

\definecolor{BBBBBB}{RGB}{0,0,255}
\definecolor{BBBBBG}{RGB}{0,100,255}
\definecolor{BBBBBR}{RGB}{100,0,255}
\definecolor{BBBBGG}{RGB}{0,150,255}
\definecolor{BBBBGR}{RGB}{100,100,255}
\definecolor{BBBBRR}{RGB}{150,0,255}
\definecolor{BBBGGG}{RGB}{0,255,255}
\definecolor{BBBGGR}{RGB}{100,150,255}
\definecolor{BBBGRR}{RGB}{150,100,255}
\definecolor{BBBRRR}{RGB}{255,0,255}
\definecolor{BBGGGG}{RGB}{0,255,150}
\definecolor{BBGGGR}{RGB}{100,255,150}
\definecolor{BBGGRR}{RGB}{150,150,150}
\definecolor{BBGRRR}{RGB}{255,100,150}
\definecolor{BBRRRR}{RGB}{255,0,150}
\definecolor{BGGGGG}{RGB}{0,255,100}
\definecolor{BGGGGR}{RGB}{100,255,100}
\definecolor{BGGGRR}{RGB}{150,255,100}
\definecolor{BGGRRR}{RGB}{255,150,100}
\definecolor{BGRRRR}{RGB}{255,100,100}
\definecolor{BRRRRR}{RGB}{255,0,100}
\definecolor{GGGGGG}{RGB}{0,255,0}
\definecolor{GGGGGR}{RGB}{100,255,0}
\definecolor{GGGGRR}{RGB}{150,255,0}
\definecolor{GGGRRR}{RGB}{255,255,0}
\definecolor{GGRRRR}{RGB}{255,150,0}
\definecolor{GRRRRR}{RGB}{255,100,0}
\definecolor{RRRRRR}{RGB}{255,0,0}

\definecolor{N}{rgb}{1,1,1}
'''
end =r'''\end{tikzpicture} 
\end{document}'''

def draw_node(i,j,color,x1,y1,x2,y2,x3,y3,x4,y4):
    tex_ts = ''
    tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j) +", " + str(-i) + ")}")
    tex_ts += "({},{}) -- ({},{}) ".format(x1,y1,x2,y2) + "{[rounded corners=9] -- " + "({},{}) -- ({},{})".format(x3,y3,x4,y4) + "} -- cycle {};\n"
    return tex_ts 


def draw_circle(i,j,color):
    tex_ts = ''
    tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j) +", " + str(-i) + ")}")
    tex_ts += "{[rounded corners=9](-0.37, -0.37) -- (0.37, -0.37)  -- (0.37,0.37) -- (-0.37,0.37) -- cycle {}};\n"
    return tex_ts 


def draw_transparent_circle(i,j,color):
    tex_ts = ''
    tex_ts += " \\fill[{},draw, shift = {} , opacity=0.0]" .format(color, "{(" + str(j) +", " + str(-i) + ")}")
    tex_ts += "{[rounded corners=9](-0.37, -0.37) -- (0.37, -0.37)  -- (0.37,0.37) -- (-0.37,0.37) -- cycle {}};\n"
    return tex_ts 


def draw_left_half_capule(i,j,color,x=0.5,y=0.37,u=0.45,v=0.37):
    return draw_node(i,j,color,x,-y,x,y,-u,v,-u,-v)


def draw_right_half_capule(i,j,color,x=0.5,y=0.37,u=0.35,v=0.37):
    return draw_node(i,j,color,-x,-y,-x,y,u,v,u,-v)


def draw_up_half_capule(i,j,color,x=0.37,y=0.5,u=0.37,v=0.45):
    return draw_node(i,j,color,-x,-y,x,-y,u,v,-u,v)


def draw_down_half_capule(i,j,color,x=0.37,y=0.5,u=0.37,v=0.45):
    return draw_node(i,j,color,-x,y,x,y,u,-v,-u,-v)


def draw_square(i,j,color):
    tex_ts = ""
    tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j) +", " + str(-i) + ")}")
    tex_ts += "( 0.5,-0.5) -- ( 0.5,0.37) {[rounded corners=9] -- (-0.35,0.37) -- (-0.35,-0.5)} -- cycle {};\n"
    tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j) +", " + str(-i) + ")}")
    tex_ts += "(-0.37, -0.5) -- (0.37, -0.5) {[rounded corners=9] -- (0.37,0.35) -- (-0.37,0.35)} -- cycle {};\n"

    tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j+1) +", " + str(-i) + ")}")
    tex_ts += "( -0.5,-0.5) -- ( -0.5,0.37) {[rounded corners=9] -- (0.35,0.37) -- (0.35,-0.5)} -- cycle {};\n"
    tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j+1) +", " + str(-i) + ")}")
    tex_ts += "(-0.37, -0.5) -- (0.37, -0.5) {[rounded corners=9] -- (0.37,0.35) -- (-0.37,0.35)} -- cycle {};\n"

    tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j) +", " + str(-i-1) + ")}")
    tex_ts += "( 0.5,-0.37) -- ( 0.5,0.5) {[rounded corners=9] -- (-0.35,0.5) -- (-0.35,-0.37)} -- cycle {};\n"
    tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j) +", " + str(-i-1) + ")}")
    tex_ts += "(-0.37, 0.5) -- (0.37, 0.5) {[rounded corners=9] -- (0.37,-0.35) -- (-0.37,-0.35)} -- cycle {};\n"

    tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j+1) +", " + str(-i-1) + ")}")
    tex_ts += "( -0.5,-0.37) -- ( -0.5,0.5) {[rounded corners=9] -- (0.35,0.5) -- (0.35,-0.37)} -- cycle {};\n"
    tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j+1) +", " + str(-i-1) + ")}")
    tex_ts += "(-0.37, 0.5) -- (0.37, 0.5) {[rounded corners=9] -- (0.37,-0.35) -- (-0.37,-0.35)} -- cycle {};\n"

    return tex_ts


def draw_conjunction(i,j,color,directions):
    tex_ts = ''
    if 'u' in directions:
        tex_ts += draw_up_half_capule(i,j,color,v=0.35)
    if 'd' in directions:
        tex_ts += draw_down_half_capule(i,j,color,v=0.35)
    if 'l' in directions:
        tex_ts += draw_left_half_capule(i,j,color,u=0.35)
    if 'r' in directions:
        tex_ts += draw_right_half_capule(i,j,color)
    return tex_ts


def draw_color_unit(tp,color,i,j,h,w):
    tex_ts = ''
    vi = i+1 < h
    vj = j+1 < w

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
    
    if tp == 1 :
        tex_ts += draw_circle(i,j,color)
    elif tp ==  10: # 4方块
        tex_ts += draw_square(i,j,color)
    elif tp == 61:
        tex_ts += draw_square(i,j,color)
        tex_ts += draw_square(i+1,j,color)
    elif tp in range(2,15) or tp in range(61,70):
        if tp == 62: tex_ts += draw_square(i-1,j,color)
        elif tp == 65: tex_ts += draw_square(i,j,color)
        locs = set([(0,0,)]+type_loc_delta[tp])
        neighbor = [(1,0),(0,1),(-1,0),(0,-1)]
        directions = ['u','l','d','r']
        ops = {
            'u':draw_up_half_capule,
            'l':draw_left_half_capule,
            'd':draw_down_half_capule,
            'r':draw_right_half_capule
        }
        for dr,dc in locs:
            marks = ''
            for (di,dj),drt in zip(neighbor,directions):
                r = dr+di
                c = dc+dj
                if (r,c,) in locs:
                    marks += drt
            if len(marks)>1:
                tex_ts += draw_conjunction(i+dr,j+dc,color,marks)
            else:
                op = ops[marks]
                tex_ts += op(i+dr,j+dc,color)
                
    elif tp == 0:
        pass
    else:
        raise ValueError('undefined type:',tp)
    return tex_ts
