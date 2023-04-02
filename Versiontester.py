import os
path = r"C:\Users\Chris\CSRNet-pytorch\0checkpoint.pth"
os.chmod(path, 0o777)
import sys
import torch
sys.path.append(os.path.abspath("C:\\Users\\Chris\\CSRNet-pytorch"))
from make_model import CSRNet

# Create an instance of the CSRNet class
model = CSRNet(load_weights=False)

# Define the path to the checkpoint file
path = r"C:\Users\Chris\CSRNet-pytorch\0checkpoint.pth"

# Load the checkpoint file
checkpoint = torch.load(path)
# Update the model's state dict

#model.load_state_dict(checkpoint['state_dict'])
