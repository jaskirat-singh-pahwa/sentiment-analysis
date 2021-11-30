import torch
import enum


class Constants(enum.Enum):
    PAD = "<PAD>"
    END = "<END>"
    UNK = "<UNK>"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
