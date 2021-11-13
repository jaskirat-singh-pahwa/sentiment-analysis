from typing import List, Tuple
import torch
import pandas as pd
from src.sentiment_analysis.preprocess import get_processed_data

pd.set_option("display.max_colwidth", 100)


def run(data_path: str) -> None:
    processed_data: List[Tuple[str, str]] = get_processed_data(data_path=data_path)
    print(processed_data[0:2])
    print(len(processed_data))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We are using {device} as device for training!")

    read_data_path: str = "/Users/jaskirat/Illinois/cs-410/TISProject/data/combined_data"
    run(data_path=read_data_path)
