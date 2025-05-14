from pathlib import Path

import matplotlib.pyplot as plt

import h5py
import numpy as np
from magtense.utils import plot_M_thin_film

from koopmag.database import create_db_mp
from koopmag.utils import plot_dynamics
from koopmag.data_utils import train_test_split
from koopmag.koopman_model import DeepKoopman

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


if __name__ == "__main__":

    datapath = Path().cwd().parent / "data"

    try:
        db = h5py.File(datapath / "120_150_40_16.h5", "r")
        # extract external fields
        Hs = np.array(db["field"])
        # extract data
        DATA = np.array(db["sequence"])
        db.close()

    except FileNotFoundError:
        print("Database not found. Please try again.")


    train_dataset, test_dataset = train_test_split(DATA, Hs, window_size=32, step=1, seed=42)