import coremltools as ct
import torch
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Convert TorchScript MobileNetV2 to Core ML')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size used in the exported model filename')
parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32', help='Precision to convert')
parser.add_argument('--dynamic-batch', action='store_true', help='Enable dynamic batch size in Core ML model')
args = parser.parse_args()

os.makedirs('models/coreml-converted', exist_ok=True)

if args.precision == 'fp32':
    model_path = f'models/coreml-converted/mobilenet_v2_b{args.batch_size}_fp32.pt'
    input_dtype = np.float32
    torch_dtype = torch.float32
    coreml_model_path = f'models/coreml-converted/mobilenet_v2_b{args.batch_size}_fp32.mlpackage'
else:
    model_path = f'models/coreml-converted/mobilenet_v2_b{args.batch_size}_fp16.pt'
    input_dtype = np.float16
    torch_dtype = torch.float16
    coreml_model_path = f'models/coreml-converted/mobilenet_v2_b{args.batch_size}_fp16.mlpackage'

# Load TorchScript model
model = torch.jit.load(model_path, map_location='cpu')
model.eval()

example_input = torch.randn(args.batch_size, 3, 224, 224, dtype=torch_dtype)

# Set up input type for Core ML conversion
if args.dynamic_batch:
    # Flexible batch size: [batch, 3, 224, 224], batch in [1, 64]
    input_type = ct.TensorType(shape=(ct.RangeDim(1, 64), 3, 224, 224), dtype=input_dtype)
else:
    input_type = ct.TensorType(shape=example_input.shape, dtype=input_dtype)

# Convert to Core ML
convert_kwargs = dict(
    model=model,
    inputs=[input_type],
    compute_units=ct.ComputeUnit.ALL,
)
# Set minimum deployment target for FP16
if args.precision == 'fp16':
    convert_kwargs['minimum_deployment_target'] = ct.target.macOS13
mlmodel = ct.convert(**convert_kwargs)
mlmodel.save(coreml_model_path)
print(f'Converted and saved {coreml_model_path}')
