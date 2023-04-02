import os
import torch
import io
from make_model import CSRNet

path = r"C:\Users\Chris\CSRNet-pytorch\partBmodel_best.pth.tar"
os.chmod(path, 0o777)

# Load the checkpoint
checkpoint = torch.load(path, map_location=torch.device('cpu'))

# Create the CSRNet model and load the state_dict
model = CSRNet(load_weights=False)
model.load_state_dict(checkpoint['state_dict'])

print("Model loaded:")
print(model)
