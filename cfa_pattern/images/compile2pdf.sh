rm hierarchy_cfa/*
python draw_hierarchy_CFA.py
cd hierarchy_cfa


# for f in *; do
#   sleep 0.5;
#   echo "compiling $f";
#   pdflatex -shell-escape \
#            --enable-write18 \
#            --extra-mem-bot=100000000 \
#            --synctex=1 $f;
# done


for f in *; do
  sleep 0.5;
  echo "compiling $f";
  lualatex $f;
done

for f in *.pdf; do
  convert -density 300 $f -quality 90 `echo $f | sed 's/\(.*\.\)pdf/\1png/'`;
done

# for f in *.png; do
#   convert -density 300 $f -quality 90 `echo $f | sed 's/\(.*\.\)png/\1jpg/'`;
# done

# rm *.aux *.log