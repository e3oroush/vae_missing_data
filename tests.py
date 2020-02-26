import torch
from modules import VAE, MNIST_DIM
from utils import (get_mnist_test_loader, get_logger, load_model, generate_syntethic_timeseries, 
generate_sinusidal_synthetic_timeseries, get_synthetic_timeseries_test_loader)
from main import reconstruct_mnist, sampling_mnist, scatter_latent_space, compute_loss_mnist, compute_loss_timeseries, reconstruct_timeseries
import numpy as np
from tqdm import tqdm
import pickle
import os
from argparse import ArgumentParser

logger = get_logger("tests")

def test_partial_data():
  model_filepath="model.pth"
  root_path="results/mnist"
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
      loss = compute_loss_mnist(x_hat, data, mu, logvar)
      test_loss += loss.item()
      z = z.cpu().numpy()
      y = y.cpu().numpy()
      zs.extend(z)
      labels.extend(y)
      reconstruct_mnist(model, data[:8], os.path.join(root_path,"partials-{}.png".format(idx)))
    test_loss /= len(test_loader.dataset)
    scatter_latent_space(np.vstack(zs), np.vstack(labels).astype("int").squeeze(), os.path.join(root_path,"distributions-partial.png"))

def test_partial_data_synthetic_ts():
  model_filepath="model-synthetic_timeseries.pth"
  root_path="results/synthetic_timeseries"
  device = torch.device("cuda")
  p=0.5
  test_loader = get_synthetic_timeseries_test_loader()
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
      loss = compute_loss_timeseries(x_hat, data, mu, logvar)
      test_loss += loss.item()
      z = z.cpu().numpy()
      y = y.cpu().numpy()
      zs.extend(z)
      labels.extend(y)
      reconstruct_timeseries(model, data[:8], os.path.join(root_path,"partials-timeseries-{}.png".format(idx)))
    test_loss /= len(test_loader.dataset)
    scatter_latent_space(np.vstack(zs), np.vstack(labels).squeeze(), os.path.join(root_path,"distributions-timeseries-partial.png"))

def generate_random_data(train_mode=True, dim=1):
  freqs = [1,2,4]
  amps = [1,0.5,0.25]
  nb_perclass=1000  if train_mode else 100
  c=0
  nb_dataset=nb_perclass*len(freqs)*len(amps)
  dataset=np.zeros((nb_dataset,dim,MNIST_DIM),dtype="float").squeeze()
  labels=np.zeros((nb_dataset,2),dtype="float")
  cnt=0
  for f in tqdm(freqs):
    for a in tqdm(amps):
      for _ in tqdm(range(nb_perclass)):
        dataset[cnt,...] = generate_sinusidal_synthetic_timeseries(a,f,dim=dim)
        cnt+=1
      labels[c*nb_perclass:(c+1)*nb_perclass,:]=[a,f]
      c+=1
  path = "data/synthetic_timeseries_{}.p".format("train" if train_mode else "test")
  logger.info("Shape of synthetic timeseries data: {} label: {}".format(dataset.shape, labels.shape))
  logger.info("Saving synthetic data to {}".format(path))
  with open(path,"wb") as fh:
    pickle.dump((dataset, labels), fh)

if __name__ == "__main__":
  parser = ArgumentParser(description="Experiments and Tests")
  parser.add_argument("--generate_timeseries_synthetic_test",help="Wether to generate test data", action="store_true", default=False)
  parser.add_argument("--generate_timeseries_synthetic_train",help="Wether to generate train data", action="store_true", default=False)
  parser.add_argument("--test_partial_data_synthetic_ts",help="Wether to run test_partial_data_synthetic_ts", action="store_true", default=False)
  parser.add_argument("--test_partial_data",help="Wether to run test_partial_data", action="store_true", default=False)
  parser.add_argument("--ts_dim", default=1, type=int, help="Dimension of synthetic timeseries")
  args = parser.parse_args()
  if args.generate_timeseries_synthetic_test:
    generate_random_data(False, args.ts_dim)
  elif args.generate_timeseries_synthetic_train:
    generate_random_data(dim=args.ts_dim)
  elif args.test_partial_data_synthetic_ts:
    test_partial_data_synthetic_ts()
  elif args.test_partial_data:
    test_partial_data()