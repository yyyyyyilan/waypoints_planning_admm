"""
train waypoints planning model
"""
#CUDA_VISIBLE_DEVICES=1 python3 main.py --enable-epsilon --mode linear --action-dim 26 
#CUDA_VISIBLE_DEVICES=1 python3 main.py --enable-epsilon --mode conv --action-dim 26 

"""
admm prune the trained model
"""
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode linear --action-dim 26 --pruning --admm
#CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode conv --action-dim 26 --pruning --admm
CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode linear --pruning --action-dim 26 --masked-retrain
#CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode conv --action-dim 26 --pruning --masked-retrain

"""
evaluate pretrained model
"""
#CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrainedd --mode linear --action-dim 26 --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --mode conv --action-dim 26 --eval

"""
evaluate pruned model
"""
#CUDA_VISIBLE_DEVICES=1 python3 main.py --load-admm-pruned --mode linear --action-dim 26 --eval
#CUDA_VISIBLE_DEVICES=1 python3 main.py --load-admm-pruned --mode conv --action-dim 26 --eval

"""
compare pretrained model vs. pruned model
"""
#CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --load-admm-pruned --mode linear --action-dim 26 --eval 
#CUDA_VISIBLE_DEVICES=1 python3 main.py --load-pretrained --load-admm-pruned --mode conv --action-dim 26 --eval