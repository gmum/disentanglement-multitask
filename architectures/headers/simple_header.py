import torch.nn as nn


class Header(nn.Module):
    def __init__(self, latent_dim, output_dim, header_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.header_dim = header_dim

        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, self.header_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.header_dim, self.header_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.header_dim, self.output_dim, bias=True)
        )

    def latent_dim(self):
        return self._latent_dim

    def output_dim(self):
        return self.output_dim

    def header_dim(self):
        return self.header_dim

    def forward(self, x):
        return self.head(x)

class DatasetHeader(nn.Module):
    def __init__(self, latent_dim, output_dim, header_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.header_dim = header_dim

        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, self.header_dim, bias=True),
            nn.Tanh(),
            nn.Linear(self.header_dim, self.header_dim, bias=True),
            nn.Tanh(),
            nn.Linear(self.header_dim, self.header_dim, bias=True),
            nn.Tanh(),
            nn.Linear(self.header_dim, self.header_dim, bias=True),
            nn.Tanh(),
            nn.Linear(self.header_dim, self.output_dim, bias=True)
        )

    def latent_dim(self):
        return self._latent_dim

    def output_dim(self):
        return self.output_dim

    def header_dim(self):
        return self.header_dim

    def forward(self, x):
        return self.head(x)
