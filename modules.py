import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import Dataset
import pickle

MNIST_DIM=784

class FCEncoder(Module):
  def __init__(self, input_dim, hidden_size, latent_size, dropout):
    super(FCEncoder, self).__init__()
    self.fc1 = nn.Linear(input_dim,hidden_size)
    self.fc_mu = nn.Linear(hidden_size, latent_size)
    self.fc_logvar = nn.Linear(hidden_size, latent_size)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    h1 = F.relu(self.fc1(self.dropout(x)))
    mu = self.fc_mu(h1)
    logvar = self.fc_logvar(h1)
    return mu, logvar

class FCDecoder(Module):
  def __init__(self, input_dim, hidden_size, latent_size, sigmoid=True):
    super(FCDecoder, self).__init__()
    self.fc1 = nn.Linear(latent_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, input_dim)
    self.sigmoid=sigmoid

  def forward(self, z):
    h1 = F.relu(self.fc1(z))
    x_hat = self.fc2(h1)
    if self.sigmoid:
      x_hat=torch.sigmoid(x_hat)
    return x_hat

class VAE(Module):
  def __init__(self, input_dim=MNIST_DIM, latent_size=2, hidden_size=400, dropout=0.5, sigmoid=True):
    super(VAE,self).__init__()
    self.latent_size = latent_size
    self.encoder = FCEncoder(input_dim, hidden_size, latent_size, dropout)
    self.decoder = FCDecoder(input_dim, hidden_size, latent_size, sigmoid=sigmoid)
    self.input_dim = input_dim
  def reparameterization(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.rand_like(std)
    return mu + eps * std
  def forward(self, x):
    mu,logvar = self.encoder(x.float().view(-1,self.input_dim)) # [batch_size,input_dim]
    z = self.reparameterization(mu,logvar)
    x_hat = self.decoder(z)
    return x_hat, mu, logvar, z

