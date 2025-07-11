export CUDA_VISIBLE_DEVICES=3
time python main_original.py \
    --root examples3 \
    --model TRELLIS \
    --skip 100 \
    --use_tracking

python rendering_video.py --result_path results_examples