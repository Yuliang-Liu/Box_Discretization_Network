export NGPUS=1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
	--config-file "configs/rects/r50_baseline.yaml" 
