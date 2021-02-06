# python test.py \
#     --opt options/test/test_MIT_config.json \
#     --network JointPixel \
#     --cfa 2JCS \
#     --pretrained_path checkpoints/2JCS/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth


python test.py \
    --opt options/test/test_MIT_config.json \
    --network JointPixel \
    --cfa RandomBlack20 \
    --pretrained_path checkpoints/RandomBlack20/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth