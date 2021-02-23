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

# for cfa in RandomFuse2 RandomFuse3 RandomFuse4
# do
#     # echo checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth
#     python test.py \
#         --opt options/test/test_MIT_config.json \
#         --network JointPixel \
#         --cfa $cfa \
#         --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth
# done

python test.py \
    --opt options/test/test_MIT_config.json \
    --network JointPixel \
    --cfa RandomFuse4 \
    --pretrained_path checkpoints/RandomFuse4/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth