# Run command
Running steps:
    1. train waypoints planning model: training the model to generate optimal selected waypoints from start to target point without obstacle collision, finally trained model will be saved in ```./saved_weights/pretrained```.
    2.  evaluate pretrinaed model: evaluate the model trained in step 1.
    3. prune the pretrained model in step 1 using ADMM: 
        - first enable ```--admm``` selecting unimportant weights in pruning config file
        - enbale ```--masked-retrain``` to mask selected unimportant waypoints to zeros
        - finally pruned model with best accuracy will be saved in ```./saved_weights/[mode]```.
    4. evaluate pruned model: evaluate the model pruned in step 3.
    5. compare pretrained model vs. pruned model: compare performance (including success rate, path, total steps, time) of pretrained model in step 1 and pruned model in step 3.


## train waypoints planning model
'''
CUDA_VISIBLE_DEVICES=1 python3 main.py --enable-epsilon --mode linear --action-dim 26 
CUDA_VISIBLE_DEVICES=1 python3 main.py --enable-epsilon --mode conv --action-dim 26 
'''


## evaluate pretrained model
'''
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode linear --action-dim 26 --eval
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode conv --action-dim 26 --eval
'''


## admm pruning the trained model
'''
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode linear --action-dim 26 --pruning --admm
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode conv --action-dim 26 --pruning --admm
'''
'''
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode linear --action-dim 26 --pruning --masked-retrain
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode conv --action-dim 26 --pruning --masked-retrain
'''


## evaluate pruned model
''
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-admm-pruned --mode linear --action-dim 26 --eval
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-admm-pruned --mode conv --action-dim 26 --eval
'''


## compare pretrained model vs. pruned model
'''
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --load-admm-pruned --mode linear --action-dim 26 --eval 
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --load-admm-pruned --mode conv --action-dim 26 --eval
'''


## Arguments
See the [[parser.py]](parser.py) file for details.