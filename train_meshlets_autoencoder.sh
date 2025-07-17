config='./configs/train-texture-autoencoder/michelangelo.yaml'
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=3
python train.py --config $config --train --gpu 0

