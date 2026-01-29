import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile
import time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_model_profile(model, input_tensor):
    original_dtype = next(model.parameters()).dtype
    model = model.float()
    input_tensor = input_tensor.float()
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor,))
    model = model.to(original_dtype)
    return macs, params
def measure_inference_time(model, input_tensor, num_runs=100):
    original_dtype = next(model.parameters()).dtype
    original_device = next(model.parameters()).device

    model = model.float().to(original_device)
    input_tensor = input_tensor.float().to(original_device)
    model.eval()

    with torch.no_grad():
        for _ in range(28):
            model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    model = model.to(original_dtype)

    avg_time = (end_time - start_time) * 1000 / num_runs # ms/count
    return avg_time

