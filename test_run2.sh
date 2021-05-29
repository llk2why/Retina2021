# b=0.0200
# for cfa in RGGB 2JCS 3JCS 4JCS Random_base RandomFuse2 RandomFuse3 RandomFuse4 RandomFuse6
# do
#     python test.py \
#         --opt options/test/test_MIT_Color.json \
#         --network JointPixel \
#         --cfa $cfa \
#         -b $b \
#         --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=$b/epochs/best_ckp.pth
# done


# b=0.0400
# for cfa in Random_base RandomBaseFuse2 RandomBaseFuse3 RandomBaseFuse4 RandomBaseFuse6
# do
#     python test.py \
#         --opt options/test/test_MIT_Sandwich.json \
#         --network JointPixel \
#         --cfa $cfa \
#         -b $b \
#         --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=$b/epochs/best_ckp.pth
# done

# 【JCS全跑通】
for a in 0.0100 0.0200
do
    # for b in 0.0000 0.0100 0.0200 0.0300 0.0400
    for b in 0.0000
    do
        for cfa in 3JCS 4JCS
        do
            CUDA_VISIBLE_DEVICES=1 python test.py \
                --opt options/test/test_jcs.json \
                --network JointPixel \
                --cfa $cfa \
                -b $b \
                -a $a \
                --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=${a}_b=$b/epochs/best_ckp.pth
        done
    done
done
