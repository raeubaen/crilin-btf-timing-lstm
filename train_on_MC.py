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
from torch.optim.lr_scheduler import _LRScheduler

from model import get_model, criterion, loss_func, save_best_model

# define device to use (cpu/gpu)
if torch.cuda.is_available():
  print('# of GPUs available: ', torch.cuda.device_count())
  print('First GPU type: ',torch.cuda.get_device_name(0))
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")


prof = cp.asarray(np.load("prof54.npy"))

prof_interp = cp.interp(cp.arange(0, 200, 0.02), cp.arange(0, 200, 0.2), prof)

plot = 0

if plot:
  plt.scatter(np.arange(0, 200, 0.02), prof_interp.get())
  plt.scatter(np.arange(0, 200, 0.2),prof.get())
  plt.show()

downsample = ndimage.zoom(prof_interp, 1e-1)
print(downsample.shape)

if plot:
  plt.scatter(np.arange(0, 200, 0.2), prof.get())
  plt.scatter(np.arange(0, 200, 0.2), downsample.get())
  plt.show()

N = 20000
k = 10

time_start = 20
seqlength = 512

mat_shifted = cp.empty((N*k, 512))

for i in range(k):
  mat = cp.tile(prof_interp, N).reshape(N, 10000)

  #r = cp.tile(cp.arange(-250, 750), int(N/1000)) #-5, +15 ns

  rows, column_indices = cp.ogrid[:mat.shape[0], :mat.shape[1]]

  r = cp.tile(cp.arange(-250, 750), int(N/1000))
  # Use always a negative shift, so that column_indices are valid.
  # (could also use module operation)
  r[r < 0] += mat.shape[1]
  column_indices = column_indices - r[:, cp.newaxis]
  del r #saving ram
  mat[:, :] = mat[rows, column_indices]

  del rows #saving ram
  del column_indices #saving ram

  mul = cp.tile(cp.expand_dims(cp.linspace(5, 45, N), (1)), 10000)
  cp.random.shuffle(mul)
  mat *= mul
  mat += cp.random.normal(0, 1.5, size=mat.shape)
  mat /= mul

  del mul #saving ram

  #512 samples from 20ns to 612/5 ns = 122.4 ns 
  mat_shifted[N*i:N*(i+1), :] = ndimage.zoom(mat, [1, 1e-1])[:, time_start*5:time_start*5+seqlength]
  del mat #saving ram

print(mat_shifted.shape)

if plot:
  plt.scatter(cp.arange(time_start*5, time_start*5+seqlength)/5., mat_shifted[3000].get())
  plt.show()

shifts = cp.tile(cp.arange(-250, 750), int(N/1000)*k)/500. #-5ns, +15ns divided by 10 to help the network -> [-0.5, 1.5]

#20ns..122.4ns divided by 100 to help the network -> [0.2, 1.224]
times = cp.tile(cp.arange(time_start*5, time_start*5+seqlength)/500., N*k).reshape(N*k, 512) 
mat_shifted = mat_shifted.reshape(N*k, 512)
data = cp.stack((mat_shifted, times), axis=2)

if plot:
  plt.scatter(data[0, :, 1], data[0, :, 0])
  plt.show()


X_train, X_test, y_train, y_test = train_test_split(data, shifts, test_size=0.1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

X_train, X_val, X_test, y_train, y_val, y_test = [
  torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_val, X_test, y_train, y_val, y_test)
]

bs = 32
jobs = 8 #8 on lxplus, 1 on colab

train_loader, val_loader, test_loader = [
  DataLoader(TensorDataset(X, y), bs, shuffle=True, num_workers=jobs) 
  for (X, y) in [(X_train, y_train), (X_val, y_val), (X_val, y_val)]
]


model = get_model()
model = model.cuda()

n_epochs = 50
iterations_per_epoch = len(train_loader)
best_acc = 0
patience, trials = 100, 0

print(model)

from torchinfo import summary
if torch.cuda.is_available():
  summary(model.cuda(), input_size=(1, 512, 2))
else:
  summary(model, input_size=(1, 512, 2))

# optimizer + lr schedular
from torch import optim
opt = optim.Adam(model.parameters(), lr=5e-4)

# scheduler for step decay lr schedule
lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[20,30], gamma=0.2)

train = 1

tloss = []
vloss = []

if train:
  print('Start model training')

  epochs = 40

  #loop over epochs
  for epoch in range(epochs):
      t0 = time.time()

      #training step
      model.train()

      train_loss = 0.0
      counter = 0

      for i, (xb, yb) in enumerate(train_loader): #takes a batch from the train dataloader 
          counter += 1
          xb=xb.type(torch.float).to(device) #move troch tensors to device (cpu or GPU)
          yb=yb.type(torch.float).to(device).unsqueeze(1)

          pred = model(xb) #get prediction for batch
          loss = loss_func(pred, yb) #compute loss

          train_loss += loss.item() #update total loss

          # backpropagation
          loss.backward()
          # update weights
          opt.step()
          # set to zero gradients for the next step
          opt.zero_grad()
          if i%20==0: print(f"\r mini-batch: {i} of {int(len(train_loader))}, loss: {loss.item()}", end="")

      train_loss = train_loss/(counter)

      tloss.append(train_loss)
      # evaluation step (same as training but w/o backpropagation)
      model.eval()

      vali_loss = 0.0

      counter = 0
      with torch.no_grad():
        for xb, yb in val_loader:
          counter += 1
          xb=xb.type(torch.float).to(device)
          yb=yb.type(torch.float).to(device).unsqueeze(1)
          pred = model(xb)
          loss = loss_func(pred, yb)
          vali_loss += loss.item()

      vali_loss = vali_loss/(counter)
      vloss.append(vali_loss)

      #save best model
      save_best_model(vali_loss, epoch, model, opt, loss_func)

      elapsed_time = time.time()-t0
      current_lr = lr_scheduler.get_last_lr()[0]
      print(
        "epoch: %d, time(s): %.2f, "
        "train loss: %.6f, "
        "vali loss: %.6f, "
        "lr : %1.2e"
          %
        (epoch+1, elapsed_time, train_loss, vali_loss, current_lr)
      )

      # update learning rate schedule
      lr_scheduler.step()

np.save("tloss", np.asarray(tloss))
np.save("vloss", np.asarray(vloss))

# load the best model
checkpoint = torch.load('./best_model.pth')
print('Best model at epoch: ', checkpoint['epoch'])

model = get_model()

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

model.to(device)
test_loss = 0.0

diffs = torch.empty(size=(0,), device=device)

timetest = torch.empty(size=(0,), device=device)

test_loss = 0.0

counter=0
with torch.no_grad():
  for xb, yb in test_loader:
    counter += 1
    xb=xb.type(torch.float).to(device)
    yb=yb.type(torch.float).to(device).unsqueeze(1)
    res = model(xb)
    tloss = loss_func(res, yb)
    diffs = torch.cat((diffs, (res-yb)), dim=0)
    timetest = torch.cat((timetest, res), dim=0)
    test_loss += tloss.item()

test_loss = test_loss/(counter)

diffs_numpy = diffs.cpu().data.numpy()
np.save("test_diffs", diffs_numpy)
print(diffs_numpy)


timetest_numpy = timetest.cpu().data.numpy()
np.save("timetest", timetest_numpy)
print(timetest_numpy)


if plot:
  plt.hist(diffs_numpy*10, bins=100)
  plt.show()

print('test loss: ', test_loss)
