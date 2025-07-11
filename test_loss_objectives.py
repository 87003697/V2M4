#!/usr/bin/env python3
"""
Test script for different loss objectives in camera estimation.
"""

import os
import subprocess
import argparse
from datetime import datetime

def run_experiment(input_dir, output_base_dir, source_images_dir, model, loss_type, use_vggt=False):
    """Run camera estimation experiment with specific loss objective."""
    
    # Create output directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"{loss_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "camera_estimation.py",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--source_images_dir", source_images_dir,
        "--model", model,
        "--loss_type", loss_type
    ]
    
    if use_vggt:
        cmd.append("--use_vggt")
    
    print(f"🚀 Running experiment: {loss_type}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 Command: {' '.join(cmd)}")
    
    # Run experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ Experiment {loss_type} completed successfully!")
            return output_dir, True, result.stdout
        else:
            print(f"❌ Experiment {loss_type} failed!")
            print(f"Error: {result.stderr}")
            return output_dir, False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {loss_type} timed out!")
        return output_dir, False, "Timeout after 1 hour"
    except Exception as e:
        print(f"💥 Experiment {loss_type} crashed: {str(e)}")
        return output_dir, False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Test multiple loss objectives')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing GLB files')
    parser.add_argument('--output_base_dir', type=str, required=True, help='Base output directory')
    parser.add_argument('--source_images_dir', type=str, required=True, help='Directory containing source images')
    parser.add_argument('--model', type=str, default='Hunyuan', choices=['TRELLIS', 'Hunyuan', 'TripoSG', 'Craftsman'])
    parser.add_argument('--loss_types', type=str, nargs='+', 
                       default=['dreamsim', 'lpips', 'l1l2', 'ssim', 'hybrid', 'mask_only'],
                       help='Loss types to test')
    parser.add_argument('--use_vggt', action='store_true', help='Use VGGT for camera search')
    
    args = parser.parse_args()
    
    print("🧪 Starting Loss Objective Comparison Experiments")
    print("=" * 60)
    
    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    results = []
    
    # Run experiments for each loss type
    for i, loss_type in enumerate(args.loss_types):
        print(f"\n📊 Testing Loss Objective: {loss_type} ({i+1}/{len(args.loss_types)})")
        print("-" * 40)
        
        output_dir, success, log = run_experiment(
            args.input_dir,
            args.output_base_dir,
            args.source_images_dir,
            args.model,
            loss_type,
            args.use_vggt
        )
        
        results.append({
            'loss_type': loss_type,
            'output_dir': output_dir,
            'success': success,
            'log': log
        })
    
    # Summary
    print("\n📊 Experiment Summary")
    print("=" * 60)
    successful_experiments = 0
    failed_experiments = 0
    
    for result in results:
        status = "✅ Success" if result['success'] else "❌ Failed"
        print(f"{result['loss_type']:12} | {status} | {result['output_dir']}")
        if result['success']:
            successful_experiments += 1
        else:
            failed_experiments += 1
    
    print(f"\nTotal: {len(results)} experiments")
    print(f"✅ Successful: {successful_experiments}")
    print(f"❌ Failed: {failed_experiments}")
    
    # Save detailed summary
    summary_path = os.path.join(args.output_base_dir, "experiment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Loss Objective Comparison Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Successful: {successful_experiments}\n")
        f.write(f"Failed: {failed_experiments}\n\n")
        
        for result in results:
            f.write(f"Loss Type: {result['loss_type']}\n")
            f.write(f"Output Dir: {result['output_dir']}\n")
            f.write(f"Success: {result['success']}\n")
            if not result['success']:
                f.write(f"Error Log: {result['log'][:500]}...\n")  # First 500 chars of error
            f.write("-" * 40 + "\n")
    
    # Save comparison script
    comparison_script_path = os.path.join(args.output_base_dir, "compare_results.py")
    with open(comparison_script_path, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""
Comparison script for loss objective results.
Generated automatically by test_loss_objectives.py
"""

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

def compare_alignment_images():
    """Compare alignment images from different loss objectives."""
    base_dir = "{args.output_base_dir}"
    
    # Find all alignment images
    alignment_images = {{}}
    for result_dir in glob.glob(os.path.join(base_dir, "*_*")):
        if os.path.isdir(result_dir):
            loss_type = os.path.basename(result_dir).split("_")[0]
            align_files = glob.glob(os.path.join(result_dir, "*_sample_mesh_align.png"))
            if align_files:
                alignment_images[loss_type] = align_files[0]  # Take first image
    
    # Create comparison plot
    if alignment_images:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (loss_type, img_path) in enumerate(alignment_images.items()):
            if i < len(axes):
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"{{loss_type}}")
                axes[i].axis('off')
        
        # Hide unused subplots
        for j in range(len(alignment_images), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "comparison.png"), dpi=150, bbox_inches='tight')
        print("Comparison saved to comparison.png")
        plt.show()
    else:
        print("No alignment images found for comparison")

if __name__ == "__main__":
    compare_alignment_images()
''')
    
    print(f"\n📋 Summary saved to: {summary_path}")
    print(f"🔍 Comparison script created: {comparison_script_path}")
    print(f"\nTo compare results visually, run:")
    print(f"cd {args.output_base_dir} && python compare_results.py")

if __name__ == "__main__":
    main() 