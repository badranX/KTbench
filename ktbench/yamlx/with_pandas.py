import csv 
import pandas as pd

from .parser import parse_dataset, parse_meta, get_feature_names_and_indent
from .writer import _dump_list, write_meta, write_dataset_heading


def _parse2dataframe(lines):
    meta, lines = parse_meta(lines) 
    features, _, _, lines = get_feature_names_and_indent(lines)
    if features:
        df = pd.DataFrame(parse_dataset(lines), columns=features)
    else:
        df = pd.DataFrame(parse_dataset(lines))
    df.attrs = meta if meta else {}
    return df

def from_yamlx(lines):
    return _parse2dataframe(lines)
    
def to_yamlx(iofile, df, is_min=True, add_column_names=True):
    write_meta(iofile, df.attrs)
    write_dataset_heading(iofile)

    df = df.reset_index(drop=True)
    features = df.columns.tolist() if add_column_names else None
    ser = df.apply(lambda x: _dump_list(x.tolist(), features, is_min, x.name), axis=1).\
        explode()
    ser.to_csv(iofile, mode='a', sep='\n', quoting= csv.QUOTE_NONE, index=False, header=False)