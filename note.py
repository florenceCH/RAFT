import os

os.system('python -u train.py --name sintel-model --stage sintel --validation sintel --gpus 0 --num_steps 10000 '
          '--batch_size 5 --lr 0.001 --image_size 200 512 --wdecay 0.0001 --mixed_precision')
# notice! raft has some trouble on scale. which means, two image_size parameters have to be a multiple of 8
os.system('python demo.py --model=checkpoints/sintel-model.pth --path=sintel-test')
