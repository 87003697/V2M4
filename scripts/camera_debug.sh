#!/bin/bash

# Camera estimation debug script with visualization
CUDA_VISIBLE_DEVICES=3 python camera_estimation.py \
    --input_dir ./results_examples/tmp \
    --output_dir ./test_visualization_final \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan \
    --loss_type dreamsim \
    --max_samples 3

echo "🎨 Visualization files generated!"
echo "📊 Check the following files:"
echo "   - *_comparison_grid.png     (Complete 2x3 grid with auto-cleanup)"
