import numpy as np
import torch
import torch.nn as nn

mse_criterion = nn.MSELoss(reduction='sum')


def train(train_loader, args, encoder, decoder):
    # switch to train mode
    encoder.train()
    decoder.train()
    for image in enumerate(train_loader):
        z = encoder(image)
        output = decoder(z)

        loss = mse_criterion(output, image) / args.batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def evaluate():
    return
