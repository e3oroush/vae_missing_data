import os
import logging
import torch
from torchvision import datasets, transforms

logger_instances = {}


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
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                  transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
  return train_loader

def get_mnist_test_loader(batch_size=128):
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
  return test_loader




def store_model(model_filepath, model):
    checkpoint={'model':model.state_dict(),
                }
    torch.save(checkpoint,model_filepath)

def load_model(model_filepath, model):
    checkpoint=torch.load(model_filepath)
    model.load_state_dict(checkpoint['model'])
    return model