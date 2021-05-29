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

# b=0.0000
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

# b=0.0000
# for cfa in RGGB
# do
#     python test.py \
#         --opt options/test/test_RGGB.json \
#         --network JointPixel \
#         --cfa $cfa \
#         -b $b \
#         --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=$b/epochs/best_ckp.pth
# done

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


# 【2021-04-20 实验】
# 【无噪模型】
# for b in 0.0000 0.0050 0.0100 0.0200 0.0300 0.0400
# do
#     for cfa in Random_base Random_pixel RandomFuse2 RandomFuse3 RandomFuse4 RandomFuse6
#     do
#         python test.py \
#             --opt options/test/test_hierarchy_clean.json \
#             --network JointPixel \
#             --cfa $cfa \
#             -b $b \
#             --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0000/epochs/best_ckp.pth
#     done
# done

# Random_base 系列
# for b in 0.0000 0.0050 0.0100 0.0200 0.0300 0.0400
# do
#     for cfa in Random_base RandomBaseFuse2 RandomBaseFuse3 RandomBaseFuse4
#     do
#         python test.py \
#             --opt options/test/test_hierarchy_base_noise.json \
#             --network JointPixel \
#             --cfa $cfa \
#             -b $b \
#             --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0400/epochs/best_ckp.pth
#     done
# done


# 【有噪模型】
# 还在跑


# 【2021-04-25 实验】

# python test.py \
#     --opt options/test/test_MIT_config.json \
#     --network JointPixel \
#     --cfa Random_pixel \
#     -b 0.0200 \
#     --pretrained_path checkpoints/Random_pixel/001_JointPixel_MIT_a=0.0000_b=0.0200/epochs/best_ckp.pth

# 【多层次有噪模型】
# b=0.0200
# for b in 0.0000 0.0050 0.0100 0.0200 0.0300 0.0400
# do
#     # for cfa in Random_pixel RandomFuse2 RandomFuse3 RandomFuse4
#     for cfa in Random_base
#     do
#         python test.py \
#             --opt options/test/test_hierarchy_b=0.0200.json \
#             --network JointPixel \
#             --cfa $cfa \
#             -b $b \
#             --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0200/epochs/best_ckp.pth
#     done
# done

# # 【JCS全跑通】
# for a in 0.0100 0.0200
# do
#     # for b in 0.0000 0.0100 0.0200 0.0300 0.0400
#     for b in 0.0000
#     do
#         for cfa in RGGB 2JCS 3JCS 4JCS
#         do
#             python test.py \
#                 --opt options/test/test_jcs.json \
#                 --network JointPixel \
#                 --cfa $cfa \
#                 -b $b \
#                 -a $a \
#                 --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=${a}_b=$b/epochs/best_ckp.pth
#         done
#     done
# done

# # # 【2021-05-01 实验】
# # # 【多层次有噪模型】
# for b in 0.0000 0.0050 0.0100 0.0200 0.0300 0.0400
# do
#     # for cfa in Random_pixel RandomFuse2 RandomFuse3 RandomFuse4
#     for cfa in Random_base Random_pixel RandomFuse2 RandomFuse3 RandomFuse4
#     do
#         python test.py \
#             --opt options/test/test_hierarchy_b=0.0100.json \
#             --network JointPixel \
#             --cfa $cfa \
#             -b $b \
#             --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0100/epochs/best_ckp.pth
#     done
# done

# 【2021-05-06 实验】
# 【多层次有噪模型】
# 【JCS 提升性能 跑通】
# for a in 0.0100 0.0200
# do
#     # for b in 0.0000 0.0100 0.0200 0.0300 0.0400
#     for b in 0.0016
#     do
#         for cfa in RGGB 2JCS 3JCS 4JCS
#         do
#             python test.py \
#                 --opt options/test/test_jcs.json \
#                 --network JointPixel \
#                 --cfa $cfa \
#                 -b $b \
#                 -a $a \
#                 --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=${a}_b=0.0000/epochs/best_ckp.pth
#         done
#     done
# done

# for b in 0.0000 0.0050 0.0100 0.0200 0.0300 0.0400
# do
#     for cfa in Random_base Random_pixel RandomFuse2 RandomFuse3 RandomFuse4
#     do
#         python test.py \
#             --opt options/test/test_hierarchy_b=0.0100.json \
#             --network JointPixel \
#             --cfa $cfa \
#             -b $b \
#             --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0100/epochs/best_ckp.pth
#     done
# done

# 【2021-05-09 实验】
# 【多层次有噪模型】
# 【JCS 提升性能 跑通】
# for a in 0.0000
# do
#     for b in 0.0016
#     do
#         for cfa in RGGB 2JCS 3JCS 4JCS
#         do
#             python test.py \
#                 --opt options/test/test_jcs.json \
#                 --network JointPixel \
#                 --cfa $cfa \
#                 -b $b \
#                 -a $a \
#                 --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=${a}_b=0.0000/epochs/best_ckp.pth
#         done
#     done
# done


# # 【2021-05-11 实验】
# # 【JCS】
# for a in 0.0300 0.0400
# do
#     for b in 0.0016
#     do
#         for cfa in RGGB 2JCS
#         do
#             python test.py \
#                 --opt options/test/test_jcs.json \
#                 --network JointPixel \
#                 --cfa $cfa \
#                 -b $b \
#                 -a $a \
#                 --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=${a}_b=0.0000/epochs/best_ckp.pth
#         done
#     done
# done

# 【2021-05-14 实验】
# for b in 0.0000 0.0050 0.0100 0.0200 0.0300 0.0400
# do
#     for cfa in Random_pixel
#     do
#         python test.py \
#             --opt options/test/test_hierarchy_b=0.0200.json \
#             --network JointPixel \
#             --cfa $cfa \
#             -b $b \
#             --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=0.0000_b=0.0200/epochs/best_ckp.pth
#     done
# done

# 【JCS】
for a in 0.0000
do
    for b in 0.0016
    do
        for cfa in RGGB
        do
            python test.py \
                --opt options/test/test_jcs.json \
                --network JointPixel \
                --cfa $cfa \
                -b $b \
                -a $a \
                --pretrained_path checkpoints/$cfa/001_JointPixel_MIT_a=${a}_b=0.0000/epochs/best_ckp.pth
        done
    done
done