import os
import logging
import torch
import timesynth as ts
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

logger_instances = {}

class SyntheticDataset(Dataset):
  def __init__(self, path):
    with open(path,"rb") as fh:
      self.x,self.y = pickle.load(fh)
  def __len__(self):
    return len(self.y)
  def __getitem__(self, index):
    return self.x[index], self.y[index]



def get_logger(logger_name, filename=None):
    """
    Returns a handy logger with both printing to std output and file
    """
    global logger_instances
    LOGGING_MODE = os.environ.get("LOGGING_MODE", 'INFO')
    log_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if logger_name not in logger_instances:
        logger_instances[logger_name] = logging.getLogger(logger_name)
        if filename is None:
            filename = os.path.join(os.environ.get(
                'LOG_DIR', '/tmp'), logger_name + ".log")

        file_handler = logging.FileHandler(filename=filename)
        file_handler.setFormatter(log_format)
        logger_instances[logger_name].addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)

        logger_instances[logger_name].addHandler(console_handler)
        logger_instances[logger_name].setLevel(
            level=getattr(logging, LOGGING_MODE))
    return logger_instances[logger_name]



def get_mnist_train_loader(batch_size=128, is_cuda=True):
  kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
  train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                  transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
  return train_loader

def get_mnist_test_loader(batch_size=128):
  test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
  return test_loader


def get_synthetic_timeseries_train_loader(batch_size=128, is_cuda=True):
  kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
  train_dataset = SyntheticDataset("data/synthetic_timeseries_train.p")
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
  return train_loader

def get_synthetic_timeseries_test_loader(batch_size=128):
  test_dataset = SyntheticDataset("data/synthetic_timeseries_test.p")
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
  return test_loader


def store_model(model_filepath, model):
    checkpoint={'model':model.state_dict(),
                }
    torch.save(checkpoint,model_filepath)

def load_model(model_filepath, model):
    checkpoint=torch.load(model_filepath)
    model.load_state_dict(checkpoint['model'])
    return model


def generate_syntethic_timeseries(ampSD=0.5,frequency=2,freqSD=0.01, n=784):
  time_sampler_pp = ts.TimeSampler(stop_time=24)
  irregular_time_samples_pp = time_sampler_pp.sample_irregular_time(num_points=2*n, keep_percentage=50)
  # Initializing Pseudoperiodic signal
  pseudo_periodic = ts.signals.PseudoPeriodic(frequency=frequency, freqSD=freqSD, ampSD=ampSD)
  # Initializing Gaussian noise
  white_noise = ts.noise.GaussianNoise(std=0.3)
  # Initializing TimeSeries class with the pseudoperiodic signal
  timeseries_pp = ts.TimeSeries(pseudo_periodic, noise_generator=white_noise)
  # Sampling using the irregular time samples
  samples_pp = timeseries_pp.sample(irregular_time_samples_pp)[0]
  # gp = ts.signals.GaussianProcess(kernel='Matern', nu=3./2)
  # gp_series = ts.TimeSeries(signal_generator=gp)
  # samples = gp_series.sample(irregular_time_samples)[0]
  return samples_pp


def generate_sinusidal_synthetic_timeseries(amp,freq,n=784,dim=1):
  x=np.tile(np.linspace(0,2*np.pi,n),(dim,1)).squeeze()
  y=amp*np.sin(x*freq)+0.2*np.random.randn(*x.shape)
  return y