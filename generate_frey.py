from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io
import string
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

#Standalone generator
#Load saved model and generate new faces
#PyTorch needs this class definition after loading
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(560, 200)
        self.fc21 = nn.Linear(200, 20)
        self.fc22 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 560)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

#Add mu, sigma so that we can read them off later
#to generate data after we disconnect the encoder
        self.mu_model = Variable(torch.cuda.FloatTensor(20), requires_grad=False)
        self.logvar_model = Variable(torch.cuda.FloatTensor(20), requires_grad=False)

        
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 560))

        self.mu_model = mu
        self.logvar_model = logvar

        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = torch.load('./out/save.model')
print(model)

mu = model.mu_model
logvar = model.logvar_model
z = model.reparametrize(mu, logvar)
x_gen = model.decode(z)


print('Generating new frey face manifold')
samples = x_gen.data.cpu().numpy()[:100]

fig = plt.figure(figsize=(10,10))
gs  = gridspec.GridSpec(10,10)
gs.update(wspace=0.01,hspace=0.01)

for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28,20), cmap='gray')

if not os.path.exists('out/'):
    os.makedirs('out/')

plt.savefig('out/generated_manifold.png',bbox_inches='tight', cmap='gray')
plt.close(fig)
