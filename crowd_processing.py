import cv2
import numpy as np
import torch
import os
import io
import pickle
import sys
import datetime
from django.utils import timezone
from PIL import Image


sys.path.append('C:\\Users\\Chris\\CSRNet-pytorch\\Crowd_Detector\\Crowd_Detector')

from make_model import CSRNet


def custom_persistent_load(saved_id):
    typename = saved_id[0]
    data = saved_id[1]

    if typename == 'torch.cuda.FloatTensor':
        storage = torch.FloatStorage.from_buffer(data)
        return storage
    else:
        raise RuntimeError(f"Unknown typename '{typename}' in persistent_load")

path = r"C:\Users\Chris\CSRNet-pytorch\1model_best.pth.tar"
os.chmod(path, 0o777)

with open(path, 'rb') as f:
    checkpoint_data = f.read()

buffer = io.BytesIO(checkpoint_data)
checkpoint = torch.load(buffer, map_location=torch.device('cpu'), pickle_module=pickle)

model = CSRNet(load_weights=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Check the model's state_dict
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def process_image(model, image):
    # Read the image directly from the file object
    image_np = np.array(Image.open(image))
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (1440, 780))
    img = img.astype(np.float32)  # Convert image to float32
    img = img / 255.0  # Scale pixel values to the range [0, 1]

    # Normalize the image using ImageNet mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = img.transpose((2, 0, 1))  # Change the channel order (H, W, C) -> (C, H, W)
    img = np.expand_dims(img, 0)    # Add an extra dimension for batch size
    img = torch.tensor(img)         # Convert the numpy array to a PyTorch tensor

    with torch.no_grad():
        density_map = model(img).detach().numpy()
        count = np.sum(density_map)
        print(f"Estimated crowd count: {count:.2f}")
        


        return count, timezone.now(), image_np







 