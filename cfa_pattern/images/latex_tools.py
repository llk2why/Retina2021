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

def draw_left_half_capule(i,j,color,x=0.5,y=0.37,u=0.45,v=0.37):
    return draw_node(i,j,color,x,-y,x,y,-u,v,-u,-v)

def draw_right_half_capule(i,j,color,x=0.5,y=0.37,u=0.35,v=0.37):
    return draw_node(i,j,color,-x,-y,-x,y,u,v,u,-v)

def draw_up_half_capule(i,j,color,x=0.37,y=0.5,u=0.37,v=0.45):
    return draw_node(i,j,color,-x,-y,x,-y,u,v,-u,v)

def draw_down_half_capule(i,j,color,x=0.37,y=0.5,u=0.37,v=0.45):
    return draw_node(i,j,color,-x,y,x,y,u,-v,-u,-v)


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
    if tp ==  2: # 横
        tex_ts += draw_left_half_capule(i,j,color)
        if vj:
            tex_ts += draw_right_half_capule(i,j+1,color)
    elif tp ==  3: # 竖
        tex_ts += draw_up_half_capule(i,j,color)
        if vi:
            tex_ts += draw_down_half_capule(i+1,j,color)
    elif tp ==  4: # 横竖 横折
        tex_ts += draw_left_half_capule(i,j,color)
        if vj:
            tex_ts += draw_conjunction(i,j+1,color,'ru')
            if vi:
                tex_ts += draw_down_half_capule(i+1,j+1,color)
    elif tp ==  5: # 竖横 竖折
        tex_ts += draw_up_half_capule(i,j,color)
        if vi:
            tex_ts += draw_conjunction(i+1,j,color,'ld')
            if vj:
                tex_ts += draw_right_half_capule(i+1,j+1,color) 
    elif tp ==  6: # 一横一竖  「
        tex_ts += draw_conjunction(i,j,color,'lu')
        if vj:
            tex_ts += draw_right_half_capule(i,j+1,color) 
        if vi:
            tex_ts += draw_down_half_capule(i+1,j,color)
    elif tp ==  7: # 」
        # 左半
        tex_ts += draw_left_half_capule(i,j-1,color)
        # 上半
        tex_ts += draw_up_half_capule(i-1,j,color)
        # 右半&下半
        tex_ts += draw_conjunction(i,j,color,'rd')
    elif tp ==  8: # ---
        # 左段
        tex_ts += draw_left_half_capule(i,j,color)
        # 中段
        tex_ts += draw_conjunction(i,j+1,color,'lr')
        # 右段
        tex_ts += draw_right_half_capule(i,j+2,color)
    elif tp ==  9: # |
        # 上段
        tex_ts += draw_up_half_capule(i,j,color)
        # 中段
        tex_ts += draw_conjunction(i+1,j,color,'ud')
        # 下段
        tex_ts += draw_down_half_capule(i+2,j,color)
    elif tp ==  10: # 4方块

        tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j) +", " + str(-i) + ")}")
        tex_ts += "( 0.5,-0.5) -- ( 0.5,0.37) {[rounded corners=9] -- (-0.35,0.37) -- (-0.35,-0.5)} -- cycle {};\n"
        tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j) +", " + str(-i) + ")}")
        tex_ts += "(-0.37, -0.5) -- (0.37, -0.5) {[rounded corners=9] -- (0.37,0.35) -- (-0.37,0.35)} -- cycle {};\n"

        if vj:
            tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j+1) +", " + str(-i) + ")}")
            tex_ts += "( -0.5,-0.5) -- ( -0.5,0.37) {[rounded corners=9] -- (0.35,0.37) -- (0.35,-0.5)} -- cycle {};\n"
            tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j+1) +", " + str(-i) + ")}")
            tex_ts += "(-0.37, -0.5) -- (0.37, -0.5) {[rounded corners=9] -- (0.37,0.35) -- (-0.37,0.35)} -- cycle {};\n"
        if vi:
            tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j) +", " + str(-i-1) + ")}")
            tex_ts += "( 0.5,-0.37) -- ( 0.5,0.5) {[rounded corners=9] -- (-0.35,0.5) -- (-0.35,-0.37)} -- cycle {};\n"
            tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j) +", " + str(-i-1) + ")}")
            tex_ts += "(-0.37, 0.5) -- (0.37, 0.5) {[rounded corners=9] -- (0.37,-0.35) -- (-0.37,-0.35)} -- cycle {};\n"
        if vj and vi:
            tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j+1) +", " + str(-i-1) + ")}")
            tex_ts += "( -0.5,-0.37) -- ( -0.5,0.5) {[rounded corners=9] -- (0.35,0.5) -- (0.35,-0.37)} -- cycle {};\n"
            tex_ts += " \\fill[{},draw, shift = {} ]" .format(color, "{(" + str(j+1) +", " + str(-i-1) + ")}")
            tex_ts += "(-0.37, 0.5) -- (0.37, 0.5) {[rounded corners=9] -- (0.37,-0.35) -- (-0.37,-0.35)} -- cycle {};\n"

    elif tp ==  11: # L
        tex_ts += draw_up_half_capule(i,j,color)
        tex_ts += draw_conjunction(i+1,j,color,'ud')
        tex_ts += draw_conjunction(i+2,j,color,'dl')
        tex_ts += draw_right_half_capule(i+2,j+1,color)
    elif tp ==  12: # |
        tex_ts += draw_up_half_capule(i,j,color)
        tex_ts += draw_conjunction(i+1,j,color,'ud')
        tex_ts += draw_conjunction(i+2,j,color,'ud')
        tex_ts += draw_down_half_capule(i+3,j,color)
    elif tp ==  13: # ----
        tex_ts += draw_left_half_capule(i,j,color)
        tex_ts += draw_conjunction(i,j+1,color,'lr')
        tex_ts += draw_conjunction(i,j+2,color,'lr')
        tex_ts += draw_right_half_capule(i,j+3,color)
    elif tp ==  14: # -|-|
        tex_ts += draw_left_half_capule(i,j,color)
        tex_ts += draw_conjunction(i,j+1,color,'ur')
        tex_ts += draw_conjunction(i+1,j+1,color,'dl')
        tex_ts += draw_right_half_capule(i+1,j+2,color)
    else:
        tex_ts += draw_circle(i,j,color)
    return tex_ts
