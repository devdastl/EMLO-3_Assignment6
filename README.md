# Lightning Template

```
copper_train --help
```

examples

- `copper_train data.num_workers=16`
- `copper_train data.num_workers=16 trainer.deterministic=True +trainer.fast_dev_run=True`

### Cat vs Dog with ViT

```
find . -type f -empty -print -delete
```

```
copper_train experiment=cat_dog data.num_workers=16 +trainer.fast_dev_run=True
```

```
copper_train experiment=cat_dog data.num_workers=16
```

## Multi Run

```
copper_train -m hydra/launcher=joblib hydra.launcher.n_jobs=4 experiment=mnist data.batch_size=8,16,32,64 data.num_workers=0
```

## Development

Install in dev mode

```
pip install -e .
```

## TODO

Workaround for `num_workers>0` in Hydra JobLib

```python
import os
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import hydra

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

@hydra.main()
def main(cfg):
    data_set = datasets.MNIST(os.getcwd(), download=True, train=True, transform=transforms.ToTensor())
    data_loader = DataLoader(data_set, batch_size=64, num_workers=cfg.num_workers, multiprocessing_context='fork')
    model = Model()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for batch in data_loader:
        x, y = batch
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
    print(f'completed {cfg.lr}')

if __name__ == '__main__':
    main()
```
