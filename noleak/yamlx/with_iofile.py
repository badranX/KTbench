from pathlib import Path

from .parser import parse_dataset, parse_meta, get_feature_names_and_indent
from .writer import write_meta
from .with_pandas import to_yamlx, from_yamlx


def _is_path(path):
    if isinstance(path, str) or isinstance(path, Path):
        return True
    return False

def decorator_open_read(func):
    def with_open(path, *args, **kwargs):
        if _is_path(path):
            with open(path, 'r') as f:
                return func(f, *args, **kwargs)
        else:
            return func(path, *args, **kwargs)
    return with_open

def decorator_open_write(func):
    def with_open(path, *args, **kwargs):
        if _is_path(path):
            with open(path, 'w') as f:
                return func(f, *args, **kwargs)
        else:
            return func(path, *args, **kwargs)
    return with_open

@decorator_open_read
def read_metadata(path, **kwargs):
    meta, _ = parse_meta(path)
    return meta

@decorator_open_write
def write_metadata(path, meta, **kwargs):
    write_meta(path, meta)
    
def read_generator(path, **kwargs):
    def gen():
        with open(path, 'r') as f:
            features, features_len, _, lines = get_feature_names_and_indent(f)
            features = features if features else list(range(features_len))
            for x in parse_dataset(lines):
                yield {f:v for f, v in zip(features, x)}
    return gen

@decorator_open_write
def write_dataframe(path, df, **kwargs):
    return to_yamlx(path, df, **kwargs)

@decorator_open_read
def read_dataframe(path, **kwargs):
    return from_yamlx(path)