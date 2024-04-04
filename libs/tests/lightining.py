from typing import Any, Dict
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import Tensor

import lightning as L
import numpy as np
import torch, os, argparse

class LitSimpleModel(L.LightningModule):
    def __init__(self, batch_size=2) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.l = nn.Linear(in_features=1, out_features=1, bias=False)
    
    def predict_step(self, batch, batch_idx) -> Any:
        idx = batch[0]
        x = batch[1][0]
        print(idx, x)
        print(f'{os.getpid()}: processing {batch_idx} of len {len(batch)} data=[{[x[0].item() for x in batch]}]')
        return idx, x[:,0]

    def on_predict_batch_end(self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        print(f'{os.getpid()}: in batch end master={self.trainer.is_global_zero}, output={type(outputs), len(outputs)}')
        idx = self.all_gather(outputs[0])
        flat_idx = torch.flatten(idx, start_dim=0, end_dim=1).squeeze()
        values = self.all_gather(outputs[1])
        flat_values = torch.flatten(values, start_dim=0, end_dim=1)
        if self.trainer.is_global_zero:
            print(f'{os.getpid()}: batch_idx={batch_idx}, flat_idx={flat_idx}')
            print(f'{os.getpid()}: saving the outputs of size {flat_values.shape} data=[{[x.item() for x in flat_values]}]')
    
    def on_predict_epoch_end(self) -> None:
        print(f'{os.getpid()}: end of epoch')


class RangeDataset(Dataset):
    def __init__(self, size=10) -> None:
        super().__init__()
        self.x = np.arange(0, size, 1, dtype=np.float32)
        print(f'{os.getpid()}: data {[n for n in self.x]}')
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, index) -> Dict[Tensor, Tensor]:
        # print(f'{os.getpid()}: getting item @ index {index} data {self.x[index][0]}')
        return torch.tensor([index]), (torch.tensor([self.x[index]]), torch.tensor([2]))
        # return torch.tensor(np.expand_dims(self.x[index], axis=0))

ds = RangeDataset(size=10)
dl = DataLoader(
    dataset=ds, 
    batch_size=2
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices", help="num of devices/processes", type=int, default=1)
    parser.add_argument(
        "--strategy", help="mp backend", type=str, default="ddp")
    parser.add_argument(
        "--accelerator", help="accelerator", type=str, default="cpu")
    
    args = parser.parse_args()
    trainer = L.Trainer(
        devices=args.devices,
        strategy=args.strategy,
        accelerator=args.accelerator
    )
    model = LitSimpleModel()
    outputs = trainer.predict(
        model=model, 
        dataloaders=dl,
    )
    print(f'{os.getpid()}: final output {len(outputs)} data=[{outputs}]')

