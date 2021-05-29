# gpu3 => tp
# ssh-copy-id -p 22 ts527.ccyen.io


# tp => gpu3
# for cfa in RandomFuse2 RandomFuse3 RandomFuse4
# do
# scp -r -P 22 llv@ts527.ccyen.io:/home/llv/Retina2021/checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0100 ./checkpoints/$cfa
# done

# for a in 0.0300 0.0400
# do
#     for cfa in 3JCS 4JCS
#     do
#     scp -r -P 22 llv@ts527.ccyen.io:/home/llv/Retina2021/checkpoints/$cfa/001_JointPixel_MIT_a=${a}_b=0.0000 ./checkpoints/$cfa
#     done
# done

for a in 0.0300 0.0400
do
    for cfa in RGGB 2JCS
    do
    scp -r -P 22 llv@ts527.ccyen.io:/home/llv/Retina2021/checkpoints/$cfa/001_JointPixel_MIT_a=${a}_b=0.0000 ./checkpoints/$cfa
    done
done