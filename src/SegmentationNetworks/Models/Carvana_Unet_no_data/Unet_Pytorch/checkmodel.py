
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
import torch

checkpoint = torch.load('my_checkpoint.pth.tar', map_location=torch.device('cpu'))
print(checkpoint)
