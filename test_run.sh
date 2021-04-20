# python test.py \
#     --opt options/test/test_MIT_config.json \
#     --network JointPixel \
#     --cfa 2JCS \
#     --pretrained_path checkpoints/2JCS/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth

# for cfa in Random_pixel Random_2JCS Random_3JCS Random_4JCS
# do
#     python test.py \
#         --opt options/test/test_MIT_config.json \
#         --network JointPixel \
#         --cfa $cfa \
#         --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth
# done

# for cfa in RandomFuse2 RandomFuse3 RandomFuse4  Random_pixel Random_2JCS Random_3JCS Random_4JCS RGGB 2JCS
# for cfa in RGGB RandomFuse4
# for cfa in 2JCS 3JCS 4JCS RandomFuse2 RandomFuse3 RandomFuse6 Random_6JCS

# for cfa in Random_base
# do
#     # echo checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth
#     python test.py \
#         --opt options/test/test_MIT_config.json \
#         --network JointPixel \
#         --cfa $cfa \
#         --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth
# done

# python test.py \
#     --opt options/test/test_MIT_config.json \
#     --network JointPixel \
#     --cfa RandomFuse4 \
#     --pretrained_path checkpoints/RandomFuse4/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth

# for cfa in RGGB 2JCS 3JCS 4JCS Random_base Random_pixel Random_2JCS Random_3JCS Random_4JCS Random_6JCS RandomFuse2 RandomFuse3 RandomFuse4 RandomFuse6
# do
#     python test.py \
#         --opt options/test/test_MIT_config.json \
#         --network JointPixel \
#         --cfa $cfa \
#         --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth
# done


# b=0.0200
# for cfa in Random_base Random_pixel RandomFuse2 RandomFuse3 RandomFuse4 RandomFuse6
# # for cfa in RandomFuse2 RandomFuse3 RandomFuse4 RandomFuse6
# # for cfa in RandomBaseFuse2 RandomBaseFuse3 RandomBaseFuse4 RandomBaseFuse6
# # for cfa in Random_base
# do
#     python test.py \
#         --opt options/test/test_MIT_Sandwich.json \
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



# 2021-04-10 实验

b=0.0000
# for cfa in Random_RandomBlack20 RGGB
# for cfa in RGGB
# do
#     python test.py \
#         --opt options/test/test_Random.json \
#         --network JointPixel \
#         --cfa $cfa \
#         -b $b \
#         --pretrained_path checkpoints/Random_RandomBlack20/001_JointPixel_MIT_a=0.0000_b=$b/epochs/best_ckp.pth
# done

b=0.0000
for cfa in RGGB
do
    python test.py \
        --opt options/test/test_RGGB.json \
        --network JointPixel \
        --cfa $cfa \
        -b $b \
        --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=$b/epochs/best_ckp.pth
done

# b=0.0000
# for cfa in RandomBlack20 Random
# do
#     python test.py \
#         --opt options/test/test_Random.json \
#         --network JointPixel \
#         --cfa $cfa \
#         -b $b \
#         --pretrained_path checkpoints/Random_RandomBlack20/001_JointPixel_MIT_a=0.0000_b=$b/epochs/best_ckp.pth
# done


