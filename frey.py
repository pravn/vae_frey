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


'''VAE for Frey faces
Take pytorch samples in https://github.com/pytorch/examples 
and adapt the VAE example to learn the frey face manifold
Code is the same except for additional hacks to 
process data, plotting and writing the generator

Heavily inspired by this version for Keras by Elvis Dohmatob:
http://dohmatob.github.io/research/2016/10/22/VAE.html'''

#Download frey faces and put them in a container dir ./frey
# wget -c http://www.cs.nyu.edu/~roweis/data.html
img_rows=28
img_cols=20
ff = scipy.io.loadmat('./frey/frey_rawface.mat')
ff = ff["ff"].T.reshape((-1, 1, img_rows, img_cols))
ff = ff.astype('float32')/255.

batch_size = 100
size = len(ff)

ff = ff[:int(size/batch_size)*batch_size]

ff_torch = torch.from_numpy(ff)

parser = argparse.ArgumentParser(description='Frey Dataset example')
args = parser.parse_args()

kwargs = {'num_workers':1, 'pin_memory':True}

train_loader = torch.utils.data.DataLoader(ff_torch, batch_size,
                                           shuffle=True, **kwargs)

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


model = VAE()
model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
#Note - MNIST had (data, labels)
#In Frey, we don't have any labels 
    for batch_idx, (data) in enumerate(train_loader):
        h_data = Variable(data)
        data = Variable(data)
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        

        if batch_idx == len(train_loader.dataset)/batch_size-1 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t"Minibatch" Loss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader),
                loss.data[0] / len(data)))


#download to host             
            samples = recon_batch.data.cpu().numpy()[:16]
            
            fig = plt.figure(figsize=(4,4))
            gs  = gridspec.GridSpec(4,4)
            gs.update(wspace=0.05, hspace=0.05)


            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28,20), cmap='gray')

            if not os.path.exists('out/'):
                os.makedirs('out/')
            
            plt.savefig('out/snapshot.png', bbox_inches='tight')
            plt.close(fig)

            samples = h_data.data.numpy()[:16]
            fig = plt.figure(figsize=(4,4))
            gs  = gridspec.GridSpec(4,4)
            gs.update(wspace=0.05, hspace=0.05)


            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28,20), cmap='gray')

            if not os.path.exists('out/'):
                os.makedirs('out/')
            
            plt.savefig('out/snapshot_o.png', bbox_inches='tight')
            plt.close(fig)

    print('====> Epoch: {} Total batch loss: {:.4f}, '.format(
          epoch, train_loss,  len(train_loader.dataset)))

    

def test(epoch):
    model.eval()
    test_loss = 0
    for data, _ in test_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


num_epochs = 100
for epoch in range(1, num_epochs + 1):
    train(epoch)

print('Done training '+str(num_epochs)+' epochs')


#save data and reload
torch.save(model, './out/save.model')





    

