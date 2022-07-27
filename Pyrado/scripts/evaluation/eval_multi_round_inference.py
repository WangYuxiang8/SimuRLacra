import pyrado
import torch

if __name__ == '__main__':
    data_real = torch.zeros(3, )
    posterior = pyrado.load(name=f"posterior.pt", load_dir="./")
    print("posterior: {0}, posterior samples: {1}".format(posterior, posterior.sample((10,), x=data_real)))
