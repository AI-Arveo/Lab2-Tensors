import torch
import numpy as np

class HousesDataset(torch.utils.data.Dataset):
    def __init__(self, train = True):
        self.dataset_size_train = 5000
        self.dataset_size_test = 20

        self.min_house_size = 30
        self.max_house_size = 70
        self.noise_house_price = 50000

        self.dataset_size = self.dataset_size_train if train else self.dataset_size_test

        house_sizes = torch.randint(self.min_house_size, self.max_house_size, (self.dataset_size,))
        house_prices = 5000 * house_sizes + 100000 + torch.randn(self.dataset_size) * self.noise_house_price

        self.data = house_sizes.float().unsqueeze(1)
        self.targets = house_prices.float().unsqueeze(1)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx, :], self.targets[idx, :]