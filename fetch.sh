# mac => jittor
ssh-copy-id -p 54147 jittor@jittor10.randonl.me


# jittor => gpu3
scp -r -P 54147 jittor@jittor00.randonl.me:/home/jittor/llv/Retina2021/checkpoints/RGGB ./checkpoints/
scp -r -P 54147 jittor@jittor00.randonl.me:/home/jittor/llv/Retina2021/logs/RGGB ./logs
