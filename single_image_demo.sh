DIR=/path/to/JPG_IMAGE_DIR
ODIR=/path/to/OUTPUT_DIR
CUDA_VISIBLE_DEVICES=0 python demo/test_single_image.py \
	--min-image-size 1000 \
	--config-file configs/r50_baseline.yaml \
	--output_dir  $ODIR\
	--img $DIR
	
