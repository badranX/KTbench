# KTbench
A knowledge tracing benchmark library. It is mainly based on Pytorch and Hugging Face datasets.

## Installing
Create a virtual environment then run the following from the project folder:
```console
pip install .
```

## Reproducing the label leakage paper

Run the following from the project folder:
```console
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
from ktbench import Pipeline, bench_model
from dataclasses import dataclass
from ktbench.model.dkt.dkt import DKT

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

## Resources
Useful tools for knowledge tracing:
- [EduStudio](https://github.com/HFUT-LEC/EduStudio): A Unified Library for Student Cognitive Modeling including Cognitive Diagnosis (CD) and Knowledge Tracing (KT). 
- [pykt-toolkit](https://github.com/pykt-team/pykt-toolkit):  A Python Library to Benchmark Deep Learning based Knowledge Tracing Models 

Some datapreprocessing and model implementations were adapted from these tools.
