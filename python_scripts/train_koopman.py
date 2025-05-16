from pathlib import Path
import sys

import matplotlib.pyplot as plt

import h5py
import numpy as np
from magtense.utils import plot_M_thin_film

from koopmag.database import create_db_mp
from koopmag.utils import plot_dynamics
from koopmag.data_utils import train_test_val_split
from koopmag.koopman_model import DeepKoopman

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


def main():

    datapath = Path().cwd().parent / "data"
    imgpath = Path().cwd().parent / "images"

    dataset_name = "1000_150_40_16.h5"

    try:
        db = h5py.File(datapath / dataset_name, "r")
        # extract external fields
        Hs = np.array(db["field"])
        # extract data
        DATA = np.array(db["sequence"])
        db.close()

    except FileNotFoundError:
        print("Database not found. Please try again.")

    print("Succefsfully loaded data.")

    batch_size = 32

    trainset, valset, _, _, _, _ = train_test_val_split(DATA, Hs, test_size=0.2, val_size=0.2, window_size=32, step=1, seed=1)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    print("Successfully created data loaders.")


    device = torch.device("cuda")

    chns = [256, 128, 64]
    latent_dim = 64
    act_fn = nn.Tanh
    learn_A = True

    koopman = DeepKoopman(
        chns=chns,
        latent_dim=latent_dim,
        act_fn=act_fn,
        learn_A=learn_A,
        ).to(device)
    
    optimizer = optim.Adam(koopman.parameters(), lr=1e-3)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    criterion = nn.MSELoss()

    epochs = 200
    valid_every = 5

    train_loss = np.zeros(epochs)
    valid_loss = []

    comment = f"{dataset_name}_Batchsize_{batch_size}_final"


    writer = SummaryWriter(log_dir=f"runs/{comment}")

    for epoch in range(epochs):

        running_loss = 0.0
        for (xtrain, utrain) in trainloader:
            
            xtrain, utrain = xtrain.to(device), utrain.to(device)

            koopman.train()
            optimizer.zero_grad()

            xhat, yhat = koopman(xtrain,  U=utrain)

            loss = criterion(xhat, xtrain[:,:-1,:,:,:]) + criterion(yhat, xtrain[:,1:,:,:,:])

            loss.backward()
            optimizer.step()
                
            running_loss += loss.item()
        
        train_loss[epoch] = running_loss / len(trainloader)

        if lr_scheduler.get_last_lr()[0] > 5e-5:
            lr_scheduler.step()

        writer.add_scalar("Loss/train", train_loss[epoch], epoch)
        writer.add_scalar("LR", lr_scheduler.get_last_lr()[0], epoch)

        if epoch % valid_every == valid_every - 1:
            koopman.eval()
            with torch.no_grad():
                running_loss_val = 0.0
                for (xtest, utest) in valloader:

                    xtest, utest = xtest.to(device), utest.to(device)

                    xhat, yhat = koopman(xtest, U=utest)

                    v_loss = criterion(xhat, xtest[:,:-1,:,:,:]) + criterion(yhat, xtest[:,1:,:,:,:])
                    running_loss_val += v_loss.item()

                valid_loss.append(running_loss_val / len(valloader))
                writer.add_scalar("Loss/valid", valid_loss[-1], epoch)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss[epoch]:.4f}, Valid Loss: {valid_loss[-1]:.4f}")

    torch.save(koopman.state_dict(), datapath / f"koopman_{comment}.pth")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_loss, label="Train loss")
    ax.plot(np.arange(valid_every - 1, epochs, valid_every), valid_loss, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid()
    plt.savefig(imgpath / f"train_val_loss_{comment}.png")


if __name__ == "__main__":
    main()

    
    

