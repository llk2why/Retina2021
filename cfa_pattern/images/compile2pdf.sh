# rm hierarchy_cfa/*
python draw_hierarchy_CFA.py
cd hierarchy_cfa

# for f in *.tex; do
#   sleep 0.5;
#   echo "compiling $f";
#   lualatex $f;
# done
lualatex Random_base.tex
lualatex Random_pixel.tex


rm *.aux *.log

# for f in *.pdf; do
#   convert -density 300 $f -quality 90 `echo $f | sed 's/\(.*\.\)pdf/\1png/'`;
# done


for f in *.pdf; do
  convert  $f[50%x50%] -quality 100% `echo $f | sed 's/\(.*\.\)pdf/\1png/'`;
done
