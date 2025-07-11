CUDA_VISIBLE_DEVICES=3 python camera_estimation.py \
    --input_dir /home/zhiyuan_ma/code2/V2M4/results_examples_w_tracking/tmp \
    --output_dir ./test_output \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan \
    --loss_type mask_only \
    --max_samples 5