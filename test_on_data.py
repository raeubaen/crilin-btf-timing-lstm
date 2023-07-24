# -*- coding: utf-8 -*-

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from cupyx.scipy import ndimage
from sklearn.model_selection import train_test_split
import time
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from model import get_model, criterion, loss_func, save_best_model

import uproot

# define device to use (cpu/gpu)
if torch.cuda.is_available():
  print('# of GPUs available: ', torch.cuda.device_count())
  print('First GPU type: ',torch.cuda.get_device_name(0))
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

plot = 0

f = uproot.open("/eos/user/r/rgargiul/www/crilin/inputML.root")

print("opened file")


charge = f["tree"]["charge"].array(library="np")

ampPeaks_chL = cp.asarray(charge[:, 0, 8])/3.72736e2 #rescaling
ampPeaks_chL = cp.tile(cp.expand_dims(ampPeaks_chL, (1)), 512)
ampPeaks_chR = cp.asarray(charge[:, 0, 9])/3.72736e2 #rescaling
ampPeaks_chR = cp.tile(cp.expand_dims(ampPeaks_chR, (1)), 512)
print("ampPeaks")

waves = f["tree"]["wave"].array(library="np")
print("read waves")

waves_chL = cp.asarray(waves[:, 0, 8, 50:562].reshape(waves.shape[0], 512))/ampPeaks_chL
waves_chR = cp.asarray(waves[:, 0, 9, 50:562].reshape(waves.shape[0], 512))/ampPeaks_chR

times = cp.asarray(f["tree"]["tWave"].array(library="np"))[:, 100:612]/100.

chL = cp.stack((waves_chL, times), axis=2)
chR = cp.stack((waves_chR, times), axis=2)

if plot:
  plt.scatter(chL[0, :, 1].get(), chL[0, :, 0].get())
  plt.show()

X_chL, X_chR = [
  torch.tensor(arr, dtype=torch.float32) for arr in (chL, chR)
]

print("got torch tensors")

bs = 32
jobs = 1 #8 on lxplus, 1 on colab

test_loader= DataLoader(TensorDataset(X_chL, X_chR), bs, num_workers=jobs)
print("loaded datasets")


# load the best model
checkpoint = torch.load('./best_model_newmodel_1500uvnoise.pth')
print('Best model at epoch: ', checkpoint['epoch'])

model = get_model()

model.load_state_dict(checkpoint['model_state_dict'])
model.eval() 


model.to(device)
test_loss = 0.0

diffs = torch.empty(size=(0,), device=device)

timeL = torch.empty(size=(0,), device=device)
timeR = torch.empty(size=(0,), device=device)

with torch.no_grad():
  for xL, xR in test_loader:
    xL=xL.type(torch.float).to(device)
    xR=xR.type(torch.float).to(device)
    resL = model(xL)
    resR = model(xR)

    diffs = torch.cat((diffs, (resL-resR)), dim=0)
    timeL = torch.cat((timeL, resL), dim=0)
    timeR = torch.cat((timeR, resR), dim=0)

diffs_numpy = diffs.cpu().data.numpy()
np.save("test_diffs_newmodel", diffs_numpy)
print(diffs_numpy)

timeL_numpy = timeL.cpu().data.numpy()
np.save("timeL_newmodel", timeL_numpy)
print(timeL)

timeR_numpy = timeR.cpu().data.numpy()
np.save("timeR_newmodel", timeR_numpy)
print(timeR)

if plot:
  plt.hist(diffs_numpy*10, bins=100)
  plt.show()
