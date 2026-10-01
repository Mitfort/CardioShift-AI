from pathlib import Path

import pandas as pd
import torch 

from src.data.dataset import ECGDataset

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "ptb"

def test_dataset() -> None:

    # Load the metadata
    df: pd.DataFrame = pd.read_csv(
        DATA_DIR / "ptbxl_prepared.csv", index_col='ecg_id'
    )

    dataset = ECGDataset(
        metadata=df,
        data_dir=DATA_DIR
    )

    # Test gettting an item from the dataset

    signal, target = dataset[0]

    assert isinstance(
        signal, 
        torch.Tensor
    )

    assert isinstance(
        target, 
        torch.Tensor
    )

    assert signal.shape == (12, 1000)

    assert target.shape == (5,)

    assert signal.dtype == torch.float32

    assert target.dtype == torch.float32

    assert torch.isfinite(signal).all()

    assert torch.isfinite(target).all()

    # Test DataLoader 

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    signals, targets = next(iter(loader))

    print(signals.shape)
    print(targets.shape)