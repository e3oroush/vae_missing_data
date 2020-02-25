import torch
from modules import VAE, MNIST_DIM
from utils import get_mnist_test_loader, get_logger, load_model
from main import reconstruct, sampling, scatter_latent_space, compute_loss
import numpy as np

def test_partial_data():
  model_filepath="model.pth"
  device = torch.device("cuda")
  p=0.5
  test_loader = get_mnist_test_loader()
  model = VAE()
  model = load_model(model_filepath, model)
  model.to(device)
  test_loss = 0
  zs = []
  labels = []
  mask_np = np.concatenate([np.zeros(MNIST_DIM//2), np.ones(MNIST_DIM//2)]).astype("int")
  with torch.no_grad():
    for idx, (data,y) in enumerate(test_loader):
      mask = torch.from_numpy(np.tile(mask_np,(len(data),1,1))).view(data.shape).long()
      mask = mask.to(device)
      data=data.to(device)
      data = data * mask # * mask.sum(-1,keepdim=True).sum(-2,keepdim=True)
      x_hat, mu, logvar, z = model(data)
      x_hat *= mask.view(-1, MNIST_DIM)
      loss = compute_loss(x_hat, data, mu, logvar)
      test_loss += loss.item()
      z = z.cpu().numpy()
      y = y.cpu().numpy()
      zs.extend(z)
      labels.extend(y)
      reconstruct(model, data[:8], "results/partials-{}.png".format(idx))
    test_loss /= len(test_loader.dataset)
    scatter_latent_space(np.vstack(zs), np.vstack(labels).astype("int").squeeze(), "results/distributions-partial.png")


if __name__ == "__main__":
    test_partial_data()