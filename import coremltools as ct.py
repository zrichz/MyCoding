import coremltools as ct
import onnx
from onnx2pytorch import ConvertModel
import torch

# Function to convert CoreML model to ONNX
def convert_coreml_to_onnx(coreml_model_path, onnx_model_path):
    # Load the CoreML model
    coreml_model = ct.models.MLModel(coreml_model_path)
    
    # Convert the CoreML model to ONNX format
    onnx_model = ct.converters.convert(coreml_model, convert_to='onnx')
    
    # Save the ONNX model
    onnx.save_model(onnx_model, onnx_model_path)
    print(f"Converted {coreml_model_path} to {onnx_model_path}")

# Function to convert ONNX model to PyTorch
def convert_onnx_to_pytorch(onnx_model_path, pytorch_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    
    # Convert the ONNX model to PyTorch format
    pytorch_model = ConvertModel(onnx_model)
    
    # Save the PyTorch model
    torch.save(pytorch_model.state_dict(), pytorch_model_path)
    print(f"Converted {onnx_model_path} to {pytorch_model_path}")

# Paths to the CoreML models
coreml_model_paths = [
    'mobileone_models_pretrained/mobileone_s0.mlmodel',
    'mobileone_models_pretrained/mobileone_s1.mlmodel',
    'mobileone_models_pretrained/mobileone_s2.mlmodel',
    'mobileone_models_pretrained/mobileone_s3.mlmodel',
    'mobileone_models_pretrained/mobileone_s4.mlmodel'
]

# Convert each CoreML model to ONNX and then to PyTorch
for coreml_model_path in coreml_model_paths:
    onnx_model_path = coreml_model_path.replace('.mlmodel', '.onnx')
    pytorch_model_path = coreml_model_path.replace('.mlmodel', '.pth')
    
    # Convert CoreML to ONNX
    convert_coreml_to_onnx(coreml_model_path, onnx_model_path)
    
    # Convert ONNX to PyTorch
    convert_onnx_to_pytorch(onnx_model_path, pytorch_model_path)