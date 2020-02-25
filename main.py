import torch
from torch.nn import Module
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from modules import VAE, MNIST_DIM
from utils import get_mnist_train_loader, get_mnist_test_loader, get_logger
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from argparse import ArgumentParser

logger = get_logger("VAE")

# Reconstruction + KL divergence losses summed over all elements and batch
def compute_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, MNIST_DIM), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def reconstruct(model: Module, data, path):
  n = len(data)
  model.eval()
  width = int(np.sqrt(MNIST_DIM))
  with torch.no_grad():
    x_hat, _, _, _ = model(data)
    image = torch.cat([data,x_hat.view(n,1,width, width)])
    save_image(image.cpu(), path, nrow=n)

def sampling(model: Module, n, path, device):
  model.eval()
  width = int(np.sqrt(MNIST_DIM))
  with torch.no_grad():
    samples = torch.randn(n, model.latent_size).to(device)
    samples = model.decoder(samples).cpu()
    save_image(samples.view(n, 1, width, width), path)

def scatter_latent_space(zs,labels,path):
  colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
  fig, ax = plt.subplots()
  for i in range(len(colors)):
    idx = np.where(labels==i)[0]
    ax.scatter([zs[i,0] for i in idx],[zs[i,1] for i in idx],c=[colors[i] for _ in idx], label=i, alpha=0.6, edgecolors="none", cmap="rainbow")
  ax.legend()
  ax.grid(True)
  fig.savefig(path, dpi=300)

def train(model: Module, device, total_epochs):
  optimizer = torch.optim.Adam(model.parameters())
  def train_epoch(epoch):
    train_loader = get_mnist_train_loader()
    model.train()
    train_loss = 0
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data, _) in pbar:
      data = data.to(device)
      optimizer.zero_grad()
      x_hat, mu, logvar, _ = model(data)
      loss = compute_loss(x_hat, data, mu, logvar)
      loss.backward()
      train_loss += loss.item()
      optimizer.step()
      # TODO tqdm progress bar
      pbar.set_description("Train Epoch: {}/{}\tLoss: {:.6f}".format(epoch,total_epochs,loss.item()/len(data)))
    logger.info("=====> Epoch: {} Average Train Loss: {:.4f}".format(epoch,train_loss/len(train_loader.dataset)))
  def test_epoch(epoch):
    test_loader = get_mnist_test_loader()
    model.eval()
    test_loss = 0
    zs = []
    labels = []
    with torch.no_grad():
      for data,y in test_loader:
        data = data.to(device)
        x_hat, mu, logvar, z = model(data)
        loss = compute_loss(x_hat, data, mu, logvar)
        test_loss += loss.item()
        z = z.cpu().numpy()
        y = y.cpu().numpy()
        zs.extend(z)
        labels.extend(y)
    test_loss /= len(test_loader.dataset)
    logger.info("=====> Epoch {} Average Test Loss: {:.4f}".format(epoch, test_loss))
    scatter_latent_space(np.vstack(zs), np.vstack(labels).astype("int").squeeze(), "results/distributions-{}.png".format(epoch))


  for epoch in range(1,total_epochs+1):
    train_epoch(epoch)
    test_epoch(epoch)
    reconstruct(model, next(iter(get_mnist_test_loader(batch_size=8)))[0].to(device), 
                      "results/reconstruction-{}.png".format(epoch))
    sampling(model, 64, "results/sampling-{}.png".format(epoch),device)
    





if __name__ == "__main__":
  parser = ArgumentParser(description="VAE example")
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
  args = parser.parse_args()
  is_cuda= not args.no_cuda
  device = torch.device("cuda" if is_cuda else "cpu")
  model = VAE().to(device)
  train(model, device, args.epochs)


