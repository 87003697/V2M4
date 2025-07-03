#!/bin/bash

# V2M4 Single Folder Processor
# Usage: ./process_single_folder.sh <image_folder_path> [model] [other_options]

set -e

# Check if folder path is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <image_folder_path> [model] [gpu_id] [other_options]"
    echo "Example: $0 examples/running_triceratops TRELLIS 2"
    echo "Available models: TRELLIS, Hunyuan, TripoSG, Craftsman"
    exit 1
fi

IMAGE_FOLDER="$1"
MODEL="${2:-TRELLIS}"
GPU_ID="${3:-0}"

# Check if folder exists
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Error: Folder $IMAGE_FOLDER does not exist!"
    exit 1
fi

# Check if folder contains images
IMAGE_COUNT=$(find "$IMAGE_FOLDER" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | wc -l)
if [ $IMAGE_COUNT -eq 0 ]; then
    echo "Error: No image files found in $IMAGE_FOLDER"
    exit 1
fi

echo "Found $IMAGE_COUNT image files in $IMAGE_FOLDER"

# Activate conda environment
echo "Activating conda environment v2m4..."
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || true
conda activate v2m4

# Check if environment is activated
if [ "$CONDA_DEFAULT_ENV" != "v2m4" ]; then
    echo "Warning: Failed to activate v2m4 environment. Current environment: $CONDA_DEFAULT_ENV"
    echo "Please manually run: conda activate v2m4"
    exit 1
fi

echo "✅ Successfully activated conda environment: $CONDA_DEFAULT_ENV"

# Set CUDA environment variables for stability
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "🔧 CUDA environment configured for GPU: $GPU_ID"

# Create temporary directory structure
TEMP_ROOT="temp_single_$(basename "$IMAGE_FOLDER")_$(date +%s)"
FOLDER_NAME=$(basename "$IMAGE_FOLDER")
TEMP_FOLDER="$TEMP_ROOT/$FOLDER_NAME"

echo "Creating temporary structure at $TEMP_ROOT..."
mkdir -p "$TEMP_FOLDER"

# Copy images to temporary structure
cp "$IMAGE_FOLDER"/*.{png,jpg,jpeg} "$TEMP_FOLDER/" 2>/dev/null || true

echo "Starting V2M4 processing with model: $MODEL on GPU: $GPU_ID"

# Run V2M4 with error handling
python main.py \
  --root "$TEMP_ROOT" \
  --output "results" \
  --model "$MODEL" \
  --N 1 \
  --n 0 \
  --skip 5 \
  --seed 42 \
  --use_vggt \
  --blender_path "blender-4.2.1-linux-x64/" \
  "${@:4}" || {
    echo "❌ Processing failed. Trying with fallback settings..."
    
    # Fallback: Try without VGGT (use DUSt3R instead)
    echo "🔄 Retrying without VGGT (using DUSt3R)..."
    python main.py \
      --root "$TEMP_ROOT" \
      --output "results" \
      --model "$MODEL" \
      --N 1 \
      --n 0 \
      --skip 8 \
      --seed 42 \
      --blender_path "blender-4.2.1-linux-x64/" \
      "${@:4}"
}

# Clean up
echo "Cleaning up temporary files..."
rm -rf "$TEMP_ROOT"

echo "Processing complete! Check results/ folder for output." 