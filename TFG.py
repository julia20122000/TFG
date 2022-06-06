#Imports necessaris
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops
import torchvision.transforms as transforms
#from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.optim import lr_scheduler
import time
import os
import copy
import torchvision
!pip install --quiet lightning-bolts
!pip install --quiet git+https://github.com/greentfrapp/lucent.git
from lucent.optvis import render, param, transform, objectives
from pl_bolts.models.self_supervised import SimCLR # per carregar el model auto-supervisat (SimCLR)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchsummary import summary
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
#Lleguir el model
resnet50 = torchvision.models.resnet50(pretrained = True, progress = True)
weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
resnet50_ss = SimCLR.load_from_checkpoint(weight_path, strict=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50.to(device).eval()
resnet50_ss.to(device).eval()
summary(resnet50, (3, 224, 224))
summary(resnet50_ss, (3, 224, 224))
#definim la funció per aconseguir la layer
def hook_PrintMaxActivated(model, input, output):
    print("Output shape:", output.shape)
    print("Layer of the model:",model)
    print("Max Activated Neuron", np.argmax(output.detach().cpu().numpy()))
def mse(img1, img2):
    err = np.square(np.subtract(img1,img2)).mean()
    return err
list_mse=[]
list_ssim=[]
for i in range(0,2047):  #mirem neurona per neurona
  matriu1 = render.render_vis(resnet50_ss, "encoder_avgpool:"+str(i), show_inline=False)
  visualitzacio = 255*matriu1[0].squeeze()
  visualitzacio = visualitzacio.astype(np.uint8)
  visualitzacio = Image.fromarray(visualitzacio)
  visualitzacio.save("encoder_avgpool"+str(i)+"self_supervied.jpg")
  img_transform = torchvision.transforms.Compose([
              torchvision.transforms.Resize(224),
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
  visualitzacio = img_transform(visualitzacio)
  visualitzacio = visualitzacio.to(device)
  visualitzacio = visualitzacio[None, :]
  #mirem la màxima activació del model auto-supervisat
  with torch.no_grad():
    handle = resnet50.avgpool.register_forward_hook(hook_PrintMaxActivated)
    out = resnet50(visualitzacio)
    handle.remove()
  max=np.argmax(out.detach().cpu().numpy())
  matriu2 = render.render_vis(resnet50, "avgpool:"+ str(max), show_inline=False)
  visualitzacio2 = 255*matriu2[0].squeeze()
  visualitzacio2 = visualitzacio2.astype(np.uint8)
  visualitzacio2 = Image.fromarray(visualitzacio2)
  visualitzacio2.save("encoder_avgpool"+str(i)+"_supervied_neuron"+str(max)+".jpg")
  img_transform = torchvision.transforms.Compose([
              torchvision.transforms.Resize(224),
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
  visualitzacio2 = img_transform(visualitzacio2)
  visualitzacio2 = visualitzacio2.to(device)
  visualitzacio2 = visualitzacio2[None, :]
  list_mse.append(mse(matriu1[0][0], matriu2[0][0]))  # mirem error MSE entre matrius
  list_ssim.append(ssim(matriu1[0][0], matriu2[0][0], multichannel=True))
np.savetxt("MSE.csv", list_mse, delimiter=',')
np.savetxt("SSIM.csv", list_ssim, delimiter=',')