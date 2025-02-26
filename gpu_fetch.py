import torch
print(torch.cuda.is_available())  # Should return True if GPU is detected
print(torch.version.cuda)  # Check installed CUDA version
print(torch.backends.cudnn.enabled)  # Should return True if cuDNN is enabled
