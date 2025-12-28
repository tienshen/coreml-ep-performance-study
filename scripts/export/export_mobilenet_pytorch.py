import torch
from torchvision import models
import argparse
import os

parser = argparse.ArgumentParser(description='Export MobileNetV2 TorchScript model')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for dummy input')
parser.add_argument('--precision', choices=['fp32', 'fp16', 'both'], default='both', help='Precision to export')
parser.add_argument('--dynamic-batch', action='store_true', help='(Note: Not supported in TorchScript export, for future use)')
args = parser.parse_args()

device = torch.device('cpu')
model = models.mobilenet_v2(pretrained=True)
model.eval()
model.to(device)

os.makedirs('models/coreml-converted', exist_ok=True)

dummy_input = torch.randn(args.batch_size, 3, 224, 224, device=device)

if args.precision in ['fp32', 'both']:
	traced_script_module = torch.jit.trace(model, dummy_input)
	traced_script_module.save(f'models/coreml-converted/mobilenet_v2_b{args.batch_size}_fp32.pt')
	print(f'Exported mobilenet_v2_b{args.batch_size}_fp32.pt to models/coreml-converted/')

if args.precision in ['fp16', 'both']:
	model_fp16 = model.half()
	dummy_input_fp16 = dummy_input.half()
	traced_script_module_fp16 = torch.jit.trace(model_fp16, dummy_input_fp16)
	traced_script_module_fp16.save(f'models/coreml-converted/mobilenet_v2_b{args.batch_size}_fp16.pt')
	print(f'Exported mobilenet_v2_b{args.batch_size}_fp16.pt to models/coreml-converted/')

if args.dynamic_batch:
	print('Note: Dynamic batch is not natively supported in TorchScript export. This flag is for future use.')
