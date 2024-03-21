# train with multi gpu
TORCH_DISTRIBUTED_DEBUG=DETAIL \
bash scripts/multi_gpu/pose_transfer_train.sh 0,1,2

# train with single gpu
# bash scripts/single_gpu/pose_transfer_train.sh 0 