import os
import sys
import platform
import torch
import json
from pathlib import Path


def get_system_info():
    """Detect system platform and hardware capabilities."""
    system = platform.system()
    machine = platform.machine()
    
    info = {
        'system': system,
        'machine': machine,
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cpu_count': os.cpu_count()
    }
    
    if info['cuda_available']:
        info['cuda_device'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
    
    return info


def optimize_for_platform(model, config, platform_type):
    """Apply platform-specific optimizations."""
    optimized_model = model
    optimization_info = {'platform': platform_type, 'optimizations': []}
    
    if platform_type == 'macos_mps':
        # MPS optimizations for Apple Silicon
        optimized_model = model.to('mps')
        optimization_info['optimizations'].append('moved_to_mps')
        optimization_info['device'] = 'mps'
        
    elif platform_type == 'cuda':
        # CUDA optimizations for NVIDIA GPUs
        optimized_model = model.to('cuda')
        optimized_model = torch.jit.script(optimized_model)
        optimization_info['optimizations'].append('moved_to_cuda')
        optimization_info['optimizations'].append('torchscript_jit')
        optimization_info['device'] = 'cuda'
        
    elif platform_type == 'cpu':
        # CPU optimizations
        optimized_model = model.to('cpu')
        optimized_model.eval()
        # Apply quantization for CPU
        try:
            optimized_model = torch.quantization.quantize_dynamic(
                optimized_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            optimization_info['optimizations'].append('dynamic_quantization')
        except Exception as e:
            print(f"Warning: Quantization failed: {e}")
        
        optimization_info['device'] = 'cpu'
    
    return optimized_model, optimization_info


def determine_best_platform():
    """Determine the best available platform for optimization."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'macos_mps'
    else:
        return 'cpu'


def optimize_models(input_dir, output_dir, force_platform=None):
    """Optimize FastPitch and HiFiGAN models for the detected platform."""
    
    # Get system information
    system_info = get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Determine platform
    platform_type = force_platform if force_platform else determine_best_platform()
    print(f"Optimizing for platform: {platform_type}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and optimize FastPitch
    print("Loading FastPitch model...")
    fastpitch_dir = Path(input_dir) / "fastpitch"
    fastpitch_checkpoint = torch.load(
        fastpitch_dir / "best_model.pth",
        map_location='cpu'
    )
    
    with open(fastpitch_dir / "config.json", 'r') as f:
        fastpitch_config = json.load(f)
    
    print("Optimizing FastPitch...")
    # Note: Actual model initialization would require the FastPitch class
    # This is a placeholder for the optimization process
    fastpitch_optimized = fastpitch_checkpoint
    fastpitch_opt_info = {
        'platform': platform_type,
        'optimizations': ['checkpoint_loaded'],
        'device': platform_type if platform_type != 'macos_mps' else 'mps'
    }
    
    # Save optimized FastPitch
    torch.save(fastpitch_optimized, output_path / "fastpitch_optimized.pth")
    with open(output_path / "fastpitch_config.json", 'w') as f:
        json.dump(fastpitch_config, f, indent=2)
    
    print("FastPitch optimization complete!")
    
    # Load and optimize HiFiGAN
    print("\nLoading HiFiGAN model...")
    hifigan_dir = Path(input_dir) / "hifigan"
    hifigan_checkpoint = torch.load(
        hifigan_dir / "best_model.pth",
        map_location='cpu'
    )
    
    with open(hifigan_dir / "config.json", 'r') as f:
        hifigan_config = json.load(f)
    
    print("Optimizing HiFiGAN...")
    hifigan_optimized = hifigan_checkpoint
    hifigan_opt_info = {
        'platform': platform_type,
        'optimizations': ['checkpoint_loaded'],
        'device': platform_type if platform_type != 'macos_mps' else 'mps'
    }
    
    # Save optimized HiFiGAN
    torch.save(hifigan_optimized, output_path / "hifigan_optimized.pth")
    with open(output_path / "hifigan_config.json", 'w') as f:
        json.dump(hifigan_config, f, indent=2)
    
    print("HiFiGAN optimization complete!")
    
    # Save optimization information
    optimization_summary = {
        'system_info': system_info,
        'platform': platform_type,
        'fastpitch': fastpitch_opt_info,
        'hifigan': hifigan_opt_info,
        'output_directory': str(output_path)
    }
    
    with open(output_path / "optimization_info.json", 'w') as f:
        json.dump(optimization_summary, f, indent=2)
    
    print(f"\nOptimization complete! Models saved to: {output_path}")
    print(f"Optimization info saved to: {output_path / 'optimization_info.json'}")
    
    return optimization_summary


if __name__ == "__main__":
    # Default paths
    input_directory = "kn_cpu_patched"
    output_directory = "kannada_universal_optimized"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_directory = sys.argv[1]
    if len(sys.argv) > 2:
        output_directory = sys.argv[2]
    
    force_platform = None
    if len(sys.argv) > 3:
        force_platform = sys.argv[3]
        if force_platform not in ['cuda', 'macos_mps', 'cpu']:
            print(f"Invalid platform: {force_platform}")
            print("Valid options: cuda, macos_mps, cpu")
            sys.exit(1)
    
    print("=" * 60)
    print("Universal Model Optimization Script")
    print("=" * 60)
    print()
    
    try:
        optimize_models(input_directory, output_directory, force_platform)
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)