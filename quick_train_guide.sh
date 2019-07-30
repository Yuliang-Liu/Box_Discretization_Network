# export NCCL_P2P_DISABLE=1
export NGPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
	--config-file "configs/r50_baseline.yaml" \
	--skip-test

