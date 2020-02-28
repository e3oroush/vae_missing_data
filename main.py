import torch
from torch.nn import Module
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from modules import VAE, MNIST_DIM
from utils import (get_mnist_train_loader, get_mnist_test_loader, get_logger, store_model,
        load_model, get_synthetic_timeseries_test_loader, get_synthetic_timeseries_train_loader,
        get_cell_timeseries_train_loader, get_cell_timeseries_test_loader)
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from argparse import ArgumentParser
import os

logger = get_logger("VAE")

# Reconstruction + KL divergence losses summed over all elements and batch
def compute_loss_mnist(recon_x, x, mu, logvar):
  BCE = F.binary_cross_entropy(recon_x, x.view(-1, MNIST_DIM), reduction='sum')

  # see Appendix B from VAE paper:
  # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
  # https://arxiv.org/abs/1312.6114
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

  return BCE + KLD

def compute_loss_timeseries(x_hat, x, mu, logvar):
  reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return reconstruction_loss + KLD

def reconstruct_mnist(model: Module, data, path):
  n = len(data)
  model.eval()
  width = int(np.sqrt(MNIST_DIM))
  with torch.no_grad():
    x_hat, _, _, _ = model(data)
    image = torch.cat([data,x_hat.view(n,1,width, width)])
    save_image(image.cpu(), path, nrow=n)

def reconstruct_timeseries(model: Module, data, path):
  n = len(data)
  model.eval()
  fig, ax = plt.subplots(n,1)
  with torch.no_grad():
    x_hat, _, _, _ = model(data)
    x_hat=x_hat.view(*data.shape)
    x_hat = x_hat.cpu().numpy()
    x = data.cpu().numpy()
    for i in range(n):
      xi = x[i][0] if len(x.shape) > 2 else x[i]
      x_hati = x_hat[i][0] if len(x.shape) > 2 else x_hat[i]
      ax[i].plot(xi,'b-')
      ax[i].plot(x_hati,'r-')
  fig.savefig(path, dpi=300)

def sampling_mnist(model: Module, n, path, device):
  model.eval()
  width = int(np.sqrt(MNIST_DIM))
  with torch.no_grad():
    samples = torch.randn(n, model.latent_size).to(device)
    samples = model.decoder(samples).cpu()
    save_image(samples.view(n, 1, width, width), path)

def scatter_latent_space(zs,labels,path):
  fcn = lambda l: '{}-{}'.format(l[0],l[1]) if type(l) == np.ndarray else l
  labels_u = np.unique([fcn(l) for l in labels])
  colors = cm.rainbow(np.linspace(0, 1, len(labels_u)))
  fig, ax = plt.subplots()
  for i in range(len(colors)):
    sel_zs = np.array([z for j,z in enumerate(zs) if fcn(labels[j]) == labels_u[i]])
    ax.scatter(sel_zs[:,0],sel_zs[:,1],c=[colors[i] for _ in sel_zs], label=labels_u[i], edgecolors="none", cmap="rainbow")
  ax.legend(loc="upper left")
  ax.grid(True)
  fig.savefig(path, dpi=300)

def train(model: Module, device, total_epochs, loss_function, get_train_loader, get_test_loader, path):
  optimizer = torch.optim.Adam(model.parameters())
  def train_epoch(epoch):
    train_loader = get_train_loader()
    model.train()
    train_loss = 0
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data, _) in pbar:
      data = data.to(device).float()
      optimizer.zero_grad()
      x_hat, mu, logvar, _ = model(data)
      loss = loss_function(x_hat.view(-1,model.input_dim), data.view(-1,model.input_dim), mu, logvar)
      loss.backward()
      train_loss += loss.item()
      optimizer.step()
      # TODO tqdm progress bar
      pbar.set_description("Train Epoch: {}/{}\tLoss: {:.6f}".format(epoch,total_epochs,loss.item()/len(data)))
    logger.info("=====> Epoch: {} Average Train Loss: {:.4f}".format(epoch,train_loss/len(train_loader.dataset)))
  def test_epoch(epoch):
    test_loader = get_test_loader()
    model.eval()
    test_loss = 0
    zs = []
    labels = []
    with torch.no_grad():
      for data,y in test_loader:
        data = data.to(device).float()
        x_hat, mu, logvar, z = model(data)
        loss = loss_function(x_hat.view(-1,model.input_dim), data.view(-1,model.input_dim), mu, logvar)
        test_loss += loss.item()
        z = z.cpu().numpy()
        y = y.cpu().numpy()
        zs.extend(z)
        labels.extend(y)
    test_loss /= len(test_loader.dataset)
    logger.info("=====> Epoch {} Average Test Loss: {:.4f}".format(epoch, test_loss))
    scatter_latent_space(np.vstack(zs), np.vstack(labels).squeeze(), os.path.join(path,"distributions-{}.png".format(epoch)))

  return train_epoch, test_epoch

def train_mnist(model: Module, device, total_epochs, path):
  train_epoch,test_epoch = train(model, device, total_epochs, 
                  compute_loss_mnist, get_mnist_train_loader, get_mnist_test_loader, path)
  for epoch in range(1,total_epochs+1):
    train_epoch(epoch)
    test_epoch(epoch)
    reconstruct_mnist(model, next(iter(get_mnist_test_loader(batch_size=8)))[0].to(device), 
                      os.path.join(path,"reconstruction-{}.png".format(epoch)))
    sampling_mnist(model, 64, os.path.join(path,"sampling-{}.png".format(epoch)),device)

def train_synthetic_timeseries(model: Module, device, total_epochs,path):
  train_epoch,test_epoch = train(model, device, total_epochs, compute_loss_timeseries, 
          get_synthetic_timeseries_train_loader, get_synthetic_timeseries_test_loader, path)
  for epoch in range(1,total_epochs+1):
    train_epoch(epoch)
    test_epoch(epoch)
    reconstruct_timeseries(model, next(iter(get_synthetic_timeseries_test_loader(batch_size=8)))[0].to(device), 
                      os.path.join(path,"timeseries-reconstruction-{}.png".format(epoch)))

def train_cell_timeseries(model: Module, device, total_epochs,path):
  train_epoch,test_epoch = train(model, device, total_epochs, compute_loss_timeseries, 
          get_cell_timeseries_train_loader, get_cell_timeseries_test_loader, path)
  for epoch in range(1,total_epochs+1):
    train_epoch(epoch)
    test_epoch(epoch)
    reconstruct_timeseries(model, next(iter(get_cell_timeseries_train_loader(batch_size=10)))[0].to(device), 
                      os.path.join(path,"timeseries-reconstruction-{}.png".format(epoch)))

if __name__ == "__main__":
  input_dims = {
    "mnist": MNIST_DIM,
    "synthetic_timeseries": MNIST_DIM,
    "cell_timeseries": 35*96
  }
  parser = ArgumentParser(description="VAE example")
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
  parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                    help='Dropout probability to set inputs zero. (default: 0.5)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
  parser.add_argument('--train_mode', type=str, default="mnist", metavar='S',
                    help='Training mode selection. Choices: mnist, synthetic_timeseries, cell_timeseries. (default: mnist)')
  args = parser.parse_args()
  model_filepath = "model-{}.pth".format(args.train_mode)
  root_path = "results/{}".format(args.train_mode)
  try:
    os.makedirs(root_path)
  except:
    pass
  is_cuda= not args.no_cuda
  device = torch.device("cuda" if is_cuda else "cpu")
  model = VAE(dropout=args.dropout, input_dim=input_dims[args.train_mode]).to(device)
  try:
    model = load_model(model_filepath, model)
    logger.info("Loading model from {}".format(model_filepath))
  except:
    logger.info("Creating VAE model from scratch")
    model = VAE(dropout=args.dropout, input_dim=input_dims[args.train_mode]).to(device)
  if args.train_mode == 'mnist':
    train_mnist(model, device, args.epochs, root_path)
  elif args.train_mode == "synthetic_timeseries":
    model.decoder.sigmoid=False # disable sigmoid from the final decoder layer
    train_synthetic_timeseries(model, device, args.epochs, root_path)
  elif args.train_mode == "cell_timeseries":
    model.decoder.sigmoid=False # disable sigmoid from the final decoder layer
    train_cell_timeseries(model, device, args.epochs, root_path)
  model.to(torch.device("cpu"))
  store_model(model_filepath, model)


