## Installation
Inside the project folder, run

```console
pip install -e .
```

## Running the paper benchmarks 

```console 
python ./baseleine/run.py
```

By default a ".ktbench" folder is created containing the experiment logs.

```
.ktbench
└── dataset_name
    └── model_name
        └── training_time_stamp
            ├── test.yaml          # Contains results on the test set.
            └── valid_fold_k.yaml  # Contains validation results on the kth fold during training.
```