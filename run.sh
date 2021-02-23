# python train.py \
#     --opt options/train/train_MIT_config.json \
#     --network JointPixel \
#     --save_image \
#     --cfa 2JCS 



# python train.py \
#     --opt options/train/train_MIT_config.json \
#     --network JointPixel \
#     --cfa 2JCS 

# for cfa in Random_2JCS Random_3JCS Random_4JCS Random_pixel
# do
#     python train.py \
#         --opt options/train/train_MIT_config.json \
#         --network JointPixel \
#         --cfa $cfa \
#         --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/last_ckp.pth
# done


for cfa in RandomFuse2 RandomFuse3 RandomFuse4
do
    python train.py \
        --opt options/train/train_MIT_config.json \
        --network JointPixel \
        --cfa $cfa 
        # --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/last_ckp.pth
done


# python train.py \
#     --opt options/train/train_MIT_config.json \
#     --network JointPixel \
#     --cfa RandomFuse4 \
#     --pretrained_path checkpoints/RandomFuse4/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/last_ckp.pth

# python train.py \
#     --opt options/train/train_MIT_config.json \
#     --network JointPixel \
#     --cfa RandomFuse4 