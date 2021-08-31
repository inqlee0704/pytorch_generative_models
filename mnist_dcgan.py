import argparse
import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from tqdm.auto import tqdm
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch

def wandb_config():
    config = wandb.config
    config.name = "DCGAN_MNIST"
    config.model = "DCGAN"
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.lrG = 0.0002
    config.lrD = 0.0002
    config.batch_size = 64
    config.b1 = 0.5
    config.b2 = 0.999

    config.img_size = 64
    config.z_size = 100

    config.debug = False
    if config.debug:
        config.epochs = 2
        config.project = "debug"
    else:
        config.epochs = 50
        config.project = "pytorch_generative_models"
    return config


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def get_accuracy(pred, label):
    pred = pred > 0.5
    return (label == pred).sum().item() / label.size(0)


class Generator(nn.Module):
    def __init__(self, latent_size=100, in_channels=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_size,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, in_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


def main():
    config = wandb_config()
    wandb.init(project=config.project, name=config.name)

    # Data Prep #
    os.makedirs("data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(config.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Loss function #
    criterion = torch.nn.BCELoss()

    # Model Prep #
    G = Generator()
    D = Discriminator()

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        criterion.cuda()

    # Initialize weights
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        G.parameters(), lr=config.lrG, betas=(config.b1, config.b2)
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(), lr=config.lrD, betas=(config.b1, config.b2)
    )

    # ----------
    #  Training
    # ----------
    fixed_noise = torch.randn(
        config.batch_size, config.z_size, 1, 1, device=config.device
    )
    iter = 0
    for epoch in range(config.epochs):
        iters = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=iters)
        for i, (real_img, _) in pbar:
            batch_size = real_img.shape[0]
            real_label = torch.full(
                (batch_size, 1, 1, 1), 1, dtype=torch.float, device=config.device
            )
            fake_label = torch.full(
                (batch_size, 1, 1, 1), 0, dtype=torch.float, device=config.device
            )
            z = torch.randn(batch_size, config.z_size, 1, 1, device=config.device)
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            gen_img = G(z)
            G_loss = criterion(D(gen_img), real_label)
            G_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            D_real_output = D(real_img.to(config.device))
            D_real_loss = criterion(D_real_output, real_label)
            D_fake_output = D(gen_img.detach())
            D_fake_loss = criterion(D_fake_output, fake_label)
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            optimizer_D.step()

            acc_real = get_accuracy(D_real_output, real_label)
            acc_fake = get_accuracy(D_fake_output, fake_label)
            accD = (acc_real + acc_fake) / 2

            pbar.set_description(
                f"Epoch {epoch} - D loss:{D_loss.item():.3f}, D acc:{accD:.3f}, G loss: {G_loss.item():.3f}"
            )
            wandb.log(
                {
                    "iteration": iter,
                    "D loss": D_loss.item(),
                    "D acc": accD,
                    "G loss": G_loss.item(),
                }
            )

            iter += 1

        with torch.no_grad():
            ref_img = G(fixed_noise).detach().cpu()
        grid_fake = vutils.make_grid(ref_img, normalize=True)
        plt.figure(figsize=(15, 15))
        plt.axis("off")
        plt.title(f"Fake Images at Epoch {epoch}")
        plt.imshow(grid_fake.permute(1, 2, 0))
        wandb.log({"plot": plt})
        plt.close()


if __name__ == "__main__":
    main()
