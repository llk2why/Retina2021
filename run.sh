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


# for cfa in RandomFuse2 RandomFuse3 RandomFuse4
# do
#     python train.py \
#         --opt options/train/train_MIT_config.json \
#         --network JointPixel \
#         --cfa $cfa 
#         # --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/last_ckp.pth
# done


# python train.py \
#     --opt options/train/train_MIT_config.json \
#     --network JointPixel \
#     --cfa Random_base \
    # -b 0.1 \
    # --cfa Random_6JCS

# python train.py \
#     --opt options/train/train_MIT_config.json \
#     --network JointPixel \
#     --cfa RandomFuse6

# for cfa in RandomFuse6 Random_6JCS 
# for cfa in RandomFuse2 RandomFuse3 RandomFuse4 RandomFuse6
# for cfa in RandomFuse4 RandomFuse6
# do
#     python train.py \
#         --opt options/train/train_MIT_config.json \
#         --network JointPixel \
#         --cfa $cfa \
#         --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/last_ckp.pth
# done

# python train.py \
#     --opt options/train/train_MIT_config.json \
#     --network JointPixel \
#     --cfa RandomFuse4 


# for cfa in RGGB 2JCS 3JCS 4JCS Random_base Random_pixel
# for cfa in RandomFuse2 RandomFuse3 RandomFuse4 RandomFuse6
# do
#     python train.py \
#         --opt options/train/train_MIT_config.json \
#         --network JointPixel \
#         --cfa $cfa \
#         -b 0.020 \
# done

# for cfa in RandomBaseFuse2 RandomBaseFuse3 RandomBaseFuse4 RandomBaseFuse6
# for cfa in Random_base
# do
#     python train.py \
#         --opt options/train/train_MIT_config.json \
#         --network JointPixel \
#         --cfa $cfa \
#         -b 0.040 \
#         # --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/last_ckp.pth \ 
# done

# for cfa in Random_RandomBlack20
# do
#     python train.py \
#         --opt options/train/train_MIT_config.json \
#         --network JointPixel \
#         --cfa $cfa \
#         -b 0.000
# done



for cfa in RGGB 2JCS 3JCS 4JCS
do
    for b in 0.0100 0.0300 0.0400
    do
        python train.py \
            --opt options/train/train_MIT_config.json \
            --network JointPixel \
            --cfa $cfa \
            -b $b
    done
done


for cfa in RGGB 2JCS 3JCS 4JCS
do
    for a in 0.0100 0.0200 0.0300 0.0400
    do
        python train.py \
            --opt options/train/train_MIT_config.json \
            --network JointPixel \
            --cfa $cfa \
            -a $a
    done
done
