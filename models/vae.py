import torch; 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
from torch.autograd import Variable


class VariationalAutoencoder(nn.Module):
    def __init__(self, conf):
        super(VariationalAutoencoder, self).__init__()

        self.encoder_layers = [len(conf["features"]) ,*conf["encoder"]]
        self.activation = conf["activation"]

        if self.activation == "relu":
            act = nn.ReLU()
        elif self.activation== "gelu":
            act = nn.GELU()
        elif self.activation== "tanh":
            act = nn.Tanh()
        elif self.activation== "selu":
            act = nn.SELU()
            
        #createing encoder
        encoder = []
        for i  in range(len(self.encoder_layers) - 1):
            dense = nn.Linear(self.encoder_layers[i],self.encoder_layers[i+1])
            batchnorm=nn.BatchNorm1d(num_features=self.encoder_layers[i+1])            

            encoder.append(dense)
            encoder.append(batchnorm)
            encoder.append(act)
            
        self.encoder = nn.Sequential(*encoder)

        #createing decoder
        self.decoder_layers = self.encoder_layers.copy()
        self.decoder_layers.reverse()
        decoder = []
        dense = nn.Linear(conf["latent_dims"],self.decoder_layers[0])

        decoder.append(dense)
        decoder.append(act)
        for i  in range(len(self.decoder_layers) - 1):
            dense = nn.Linear(self.decoder_layers[i],self.decoder_layers[i+1])
            batchnorm=nn.BatchNorm1d(num_features=self.decoder_layers[i+1]) 

            decoder.append(dense)
            decoder.append(batchnorm)
            decoder.append(act)
            
        self.decoder = nn.Sequential(*decoder)

        # creating laten sapace
        self.mu = nn.Linear(self.encoder_layers[-1], conf["latent_dims"])
        self.sigma = nn.Linear(self.encoder_layers[-1], conf["latent_dims"])

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        
        mu =  self.mu(x)
        sigma = self.sigma(x)
        z = self.reparameterize(mu, sigma)
        self.kl = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

        decoder = self.decoder(z)
        return decoder, z
    


    


def train(autoencoder, data, device, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat,_ = autoencoder(x.float())
            loss = ((x - x_hat)**2).sum() + autoencoder.kl
            loss.backward()
            opt.step()
            
        print(loss.item())
    return autoencoder


