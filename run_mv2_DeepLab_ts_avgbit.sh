#!/bin/bash
batch='64' # in NIPQ 256
date='0808'
lr='0.04' # in deeplab --lr 0.01 
lr_policy='cosine_warmup' # in deeplab poly
decay='1e-5'
total_itrs='60'
# epoch='40' In DeepLabv3 total iter

model='deeplabv3plus_mobilenet_nq' # in NIPQ 'mobilenetv2'
dataset='voc' # in NIPQ 'imagenet'

a_scale='1'
w_scale='1'
bops_scale='3'

mode='avgbit'
target='8'


CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main_nq.py \
        --model $model --mode $mode --target $target --total_itrs $total_itrs --bops_scale ${bops_scale} \
        --gpu_id 0,1,2,3 --output_stride 16 \
        --warmuplen 3 --ft_epoch 6 --lr_policy $lr_policy \
        --ckpt '/nas3/0.Personal/seunghun/DeepLabV3Plus-Pytorch/checkpoints/baseline_deeplabv3plus_mobilenet_voc_os16.pth' \
        --dataset $dataset --year 2012_aug --crop_val --lr $lr --weight_decay $decay  --crop_size 513 \
        --batch_size $batch --a_scale $a_scale --w_scale $w_scale \
        --ckpt_path ./checkpoint/${date}/${lr_policy}/${model}_${dataset}_${mode}_${total_itrs}_${a_scale}_${w_scale}_target_${target} \
        # >& ./log/${model}_${dataset}_${mode}_${epoch}_${a_scale}_${w_scale}_target_${target}.log &