import torch
import torchvision
import torch.nn as nn

model_ft = torchvision.models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

PATH = r'E:\Estek-AIProject\INTEL-SOOP\PyTorch\Model3\model3.pt'

model_ft.load_state_dict(torch.load(PATH))

x = torch.randn(1,3,224,224,requires_grad=False)
torch_out = torch.onnx._export(model_ft,x,r"E:\Estek-AIProject\INTEL-SOOP\PyTorch\Model3\Model3.onnx",export_params=True)