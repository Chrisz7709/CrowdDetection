import torch
import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(cwd, '..', 'CSRNet_pytorch')))



from Crowd_Detector.model import CSRNet


def load_csrnet_model(model_path):
    model = CSRNet()
    model = model.cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

csrnet_model_path = os.path.join(r'C:\Users\Chris\CSRNet-pytorch\Crowd_Detector\Crowd_Detector', '1model_best.pth.tar')
csrnet_model = load_csrnet_model(csrnet_model_path)
