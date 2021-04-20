export CUDA_VISIBLE_DEVICES=1
b=0.0000
for cfa in Random_2JCS Random_3JCS Random_4JCS Random_6JCS RandomFuse2 RandomFuse3 RandomFuse4 RandomFuse6
do
    python test.py \
        --opt options/test/test_MIT_config.json \
        --network JointPixel \
        --cfa $cfa \
        -b $b \
        --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=$b/epochs/best_ckp.pth
done
