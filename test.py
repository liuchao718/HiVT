import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda) # 查看CUDA的版本号