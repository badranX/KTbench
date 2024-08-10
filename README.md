## Installing, Training, and Evaluating the models from the paper

Create a virtual environment then run the following from the project folder:
```console
pip install -r requirements.txt
pip install -e .
python baseline/run.py
```

By default a ".ktbench" folder is created, containing the experiment logs:

```
.ktbench
└── dataset_name
    └── model_name
        └── training_time_stamp
            ├── test.yaml          # Contains results on the test set.
            └── valid_fold_k.yaml  # Contains validation results on the kth fold during training.
```

## Usage
An example of training and evaluating a DKT with basic KC-expanded sequence
```python
from noleak import Pipeline, bench_model
from dataclasses import dataclass
from noleak.model.dkt.dkt import DKT

@dataclass
class Cfg:
    model_cls = DKT
    window_size: int = 150
    is_unfold = True
    eval_method = Pipeline.EVAL_UNFOLD_KC_LEVEL
    kfold = 5

@dataclass
class Traincfg:
    batch_size = 128
    eval_batch_size = 128
    n_epoch = 100
    lr = 0.001

bench_model(Cfg(), Traincfg(), datasets = ['assist2009'])

```

