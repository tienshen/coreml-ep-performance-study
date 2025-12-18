#!/usr/bin/env python3
"""Script to export torchvision models to ONNX format."""

import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torchvision.models as models

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.models import ensure_models_dir


def export_vision_model(
    model_name: str,
    batch_size: int,
    height: int,
    width: int,
    fp16: bool = False,
    opset_version: int = 14,
    output_path: Path = None
):
    """Export a torchvision model to ONNX format."""
    
    print(f"Exporting {model_name} to ONNX...")
    
    # Load model
    print(f"Loading {model_name} from torchvision...")
    try:
        model_fn = getattr(models, model_name)
        # Use new weights API (torchvision >= 0.13)
        if model_name == "mobilenet_v2":
            model = model_fn(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif model_name == "mobilenet_v3_large":
            model = model_fn(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        elif model_name == "mobilenet_v3_small":
            model = model_fn(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            # Generic fallback
            model = model_fn(pretrained=True)
    except AttributeError:
        print(f"✗ Model '{model_name}' not found in torchvision.models")
        print(f"Available models: mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small")
        sys.exit(1)
    
    model.eval()
    print("Model loaded successfully")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")
    
    # Convert to FP16 if requested
    if fp16:
        print("Converting model to FP16...")
        model = model.half()
    
    # Create dummy input (ImageNet standard: 3 channels, HxW)
    print(f"Exporting to ONNX: {output_path}")
    dtype = torch.float16 if fp16 else torch.float32
    dummy_input = torch.randn(batch_size, 3, height, width, dtype=dtype)
    
    # Dynamic or static shapes
    dynamic_axes = None if batch_size > 0 else {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    input_names = ['input']
    output_names = ['output']
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"Model exported successfully to {output_path}")
    
    # Print model info
    import onnx
    onnx_model = onnx.load(str(output_path))
    print(f"\nONNX Model Info:")
    print(f"  Opset version: {onnx_model.opset_import[0].version}")
    print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
    
    # Check file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export torchvision models to ONNX format"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Torchvision model name (e.g., 'mobilenet_v2', 'mobilenet_v3_large')"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for export (default: 1, use 0 for dynamic)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Input image height (default: 224)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Input image width (default: 224)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export model in FP16 precision"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Generate output filename
    if args.output_name:
        output_name = args.output_name
    else:
        suffix = f"_b{args.batch}_h{args.height}_w{args.width}"
        suffix += "_fp16" if args.fp16 else "_fp32"
        output_name = args.model_name + suffix
    
    if not output_name.endswith(".onnx"):
        output_name += ".onnx"
    
    # Ensure models directory exists
    models_dir = ensure_models_dir()
    output_path = models_dir / output_name
    
    print(f"Output path: {output_path}")
    print()
    
    try:
        export_vision_model(
            model_name=args.model_name,
            batch_size=args.batch,
            height=args.height,
            width=args.width,
            fp16=args.fp16,
            opset_version=args.opset_version,
            output_path=output_path
        )
        print("\n✓ Export completed successfully")
    except Exception as e:
        print(f"\n✗ Error exporting model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
