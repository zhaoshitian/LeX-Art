#!/usr/bin/env sh

train_data_path='./configs/10k_text.yaml'

model=NextDiT_2B_GQA_patch2_Adaln_Refiner
# check_path=path_to_existing_dir/checkpoints/00xxxxx
check_path=path_to_checkpoint_of_Lumina-Image-2.0
batch_size=256
micro_size=8
snr_type=lognorm
lr=1e-4
precision=bf16
size=1024 # resolution
save_interval=1000

exp_name=${model}_bs${batch_size}_lr${lr}_${precision}

prefix=path_to_save_predix_dir
save_dir="${prefix}/10k/${exp_name}"
mkdir -p ${save_dir}

NNODES=1
NPROC_PER_NODE=8
MASTER_PORT=1234 #1234
NODE_RANK=0

export WORLD_SIZE=$(($NPROC_PER_NODE*$NNODES))
export LOCAL_RANK=${NODE_RANK:-0}
export OMP_NUM_THREADS=1

python -m torch.distributed.run \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr "localhost" \
    --master_port ${MASTER_PORT} \
    finetune.py \
    --global_bsz_${size} ${batch_size} \
    --micro_bsz_${size} ${micro_size} \
    --model ${model} \
    --lr ${lr} --grad_clip 2.0 \
    --data_path ${train_data_path} \
    --results_dir ${save_dir} \
    --data_parallel fsdp \
    --max_steps 15000 \
    --ckpt_every ${save_interval} \
    --log_every 1 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --global_seed 20250217 \
    --num_workers 12 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --checkpointing \
    --init_from ${check_path} \
    2>&1 | tee -a ${save_dir}/log.txt